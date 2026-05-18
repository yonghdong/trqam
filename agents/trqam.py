import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class TRQAMAgent(flax.struct.PyTreeNode):
    """Trust Region Q-learning with Adjoint Matching (TRQAM)."""

    rng: Any
    network: Any
    lam: float  # Trust-region parameter λ
    kl_ema: float  # EMA-smoothed path-space KL estimate
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        total_loss = critic_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def get_lambda(self):
        """Return the trust-region parameter λ used in the SOC dynamics."""
        
        return self.config["lam_scale"] * self.lam

    def get_sigma_scale(self):
        """Compute 1/√λ for the adjoint-matching loss."""

        return 1.0 / jnp.sqrt(jnp.maximum(self.get_lambda(), 1e-8))

    @partial(jax.jit, static_argnames=("flow_steps",))
    def adj_matching(self, obs, rng, flow_steps=None):
        """Adjoint matching for flow policy fine-tuning.

        Forward pass uses OT memoryless Euler discretization:
        X_{t+h} = X_t + h(2v(X_t,t) - X_t/t) + sqrt(h)*g(t)*ε,  ε ~ N(0,I)

        where g(t) = sqrt(2(1-t)/t) is the memoryless noise schedule.
        """
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps

        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")

        h = 1 / flow_steps
        xs = [x]
        ts = []

        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            g_t = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)

            if i != flow_steps - 1:
                v_fast = self.network.select("actor_fast")(obs, x, t)
                x = x + h * (2 * v_fast - x / (t + h)) + jnp.sqrt(h) * g_t * noise
            else:
                x = x + h * actor_slow(obs, x, t)

            xs.append(x)
            ts.append(t)

        # Compute the critic's action gradient as the adjoint state initialization
        # From the algorithm: ã_1 = -∇_{x_1} Q(s, X_1) (no temperature scaling)
        critic_network = "target_critic" if self.config["use_target_grad"] else "critic"
        if self.config['clip_adj']:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, jnp.clip(y, -1., 1.)).mean(axis=0).sum(), 1)
        else:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, y).mean(axis=0).sum(), 1)

        adj = -grad_fn(obs, xs[-1])
        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
        }
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)

            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp

            adjs.append(adj)

        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, adj_rng = jax.random.split(rng, 4)

        ## BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss

        info = {}
        total_fast_loss = 0
        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")

        # Skip adjoint matching if bc_only mode
        if self.config.get("bc_only", False):
            return actor_loss, {'flow_loss': flow_loss, "fast_loss": 0.0}

        ## Adjoint matching with the current trust-region λ
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        h = 1 / self.config["flow_steps"]

        # g(t)² = 2(1-t)/t, σ(t) = g(t)/√λ for the adjoint-matching loss
        sigma_scale = self.get_sigma_scale()
        g_t_sq = 2 * (1 - ts + h) / (ts + h)
        g_t = jnp.sqrt(g_t_sq)
        sigmas = sigma_scale * g_t

        observations = jnp.repeat(batch["observations"][None], self.config["flow_steps"], axis=0)
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)
        vf_base = actor_slow(observations, xs, ts)

        # Path-space KL: Σ_k (2h/g(t_k)²) * ||v_fin - v_base||²
        # = Σ_k (h/(2g²)) * ||2(v_fin - v_base)||²
        vel_diff = vf_fine - vf_base
        vel_diff_sq = jnp.sum(vel_diff ** 2, axis=-1)  # (flow_steps, batch_size)
        kl_per_step = (2 * h / g_t_sq[..., 0]) * vel_diff_sq
        path_kl = jnp.mean(jnp.sum(kl_per_step, axis=0))  # sum over steps, mean over batch
        # Normalize by action horizon
        horizon = self.config['horizon_length'] if self.config["action_chunking"] else 1
        path_kl = path_kl / horizon

        # Adjoint matching loss
        adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)
        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info["adj_loss"] = adj_loss
        info["path_kl"] = path_kl
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        return actor_loss + total_fast_loss, {'flow_loss': flow_loss, "fast_loss": total_fast_loss, **info}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Skip critic loss in bc_only mode
        if self.config.get("bc_only", False):
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            return actor_loss, info

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def dual_update(self, kl_estimate):
        """Projected dual descent update for the trust-region parameter λ."""

        # Safety: clip KL estimate to prevent explosions
        kl_clip_max = self.config["kl_clip_coef"] * self.config["kl_budget"]
        kl_estimate_clipped = jnp.minimum(kl_estimate, kl_clip_max)

        # EMA smoothing: D = (1-ρ)*D_prev + ρ*D̂
        new_kl_ema = (1 - self.config["kl_ema_coef"]) * self.kl_ema + \
                     self.config["kl_ema_coef"] * kl_estimate_clipped

        # Projected dual descent: raise λ when KL exceeds budget, lower it otherwise
        constraint_violation = new_kl_ema - self.config["kl_budget"]
        new_lam = jnp.maximum(
            self.config["lambda_min"],
            self.lam + self.config["eta_lambda"] * constraint_violation
        )
        new_lam = jnp.minimum(new_lam, self.config["lambda_max"])

        return new_lam, new_kl_ema

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)

        # Skip target updates and dual update in bc_only mode
        if agent.config.get("bc_only", False):
            return agent.replace(
                network=new_network,
                rng=new_rng,
            ), info

        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor_slow')

        # Path KL estimate from actor loss
        kl_estimate = info['actor/path_kl']

        # Projected dual descent on the trust-region parameter λ
        new_lam, new_kl_ema = agent.dual_update(kl_estimate)

        # Log trust-region diagnostics
        lambda_value = agent.config["lam_scale"] * new_lam
        sigma_scale = 1.0 / jnp.sqrt(jnp.maximum(lambda_value, 1e-8))
        kl_clip_max = agent.config["kl_clip_coef"] * agent.config["kl_budget"]
        info['dual/lambda'] = lambda_value
        info['dual/sigma_scale'] = sigma_scale
        info['dual/kl_estimate'] = kl_estimate
        info['dual/kl_estimate_clipped'] = jnp.minimum(kl_estimate, kl_clip_max)
        info['dual/kl_clipped'] = (kl_estimate > kl_clip_max).astype(jnp.float32)  # 1 if clipped
        info['dual/kl_ema'] = new_kl_ema
        info['dual/kl_budget'] = agent.config["kl_budget"]

        return agent.replace(
            network=new_network,
            rng=new_rng,
            lam=new_lam,
            kl_ema=new_kl_ema
        ), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng,
    ):
        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)

        # Use the fine-tuned flow policy
        actions = self.compute_flow_actions(observations, noises, model="fast")
        actions = jnp.clip(actions, -1, 1)

        # best-of-n sampling
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @partial(jax.jit, static_argnames="model")
    def compute_flow_actions(
        self,
        observations,
        noises,
        model="slow",
    ):
        actions = noises
        networks = [self.network.select(f'actor_{m}') for m in model.split(",")]

        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum([network(observations, actions, t) for network in networks])

            actions = actions + vels / self.config['flow_steps']

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            action_dim=full_action_dim,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),  # clip grad norm
                optax.adam(learning_rate=config["lr"]))
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        # Initialize trust-region parameter λ and KL EMA
        initial_lam = 1.0
        initial_kl_ema = 0.0

        return cls(
            rng,
            network=network,
            lam=initial_lam,
            kl_ema=initial_kl_ema,
            config=flax.core.FrozenDict(**config)
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='trqam',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).

            ## Common hyperparamters
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,

            ## Q-chunking hyperparameters
            horizon_length=ml_collections.config_dict.placeholder(int), # Will be set
            action_chunking=False,                                      # Use Q-chunking or just n-step return

            ## RL hyperparameters
            num_qs=10,      # Critic ensemble size
            rho=0.5,        # Pessimistic backup

            discount=0.995,  # Discount factor.
            tau=0.005,      # Target network update rate.
            flow_steps=10,  # Number of flow steps.

            best_of_n=1,    # Best-of-n for computing Q-targets and sampling actions.

            ## Trust-region hyperparameters
            lam_scale=3.0,          # Fixed multiplicative scale applied to λ in σ(τ)
            kl_budget=1.0,          # KL budget ε_KL (target path-space KL)
            eta_lambda=0.01,        # Dual descent step size η_λ
            kl_ema_coef=0.1,        # EMA coefficient ρ for KL smoothing
            lambda_min=0.01,        # Lower bracket on λ
            lambda_max=100.0,       # Upper bracket on λ
            kl_clip_coef=2.0,       # KL clip coefficient (clip at kl_clip_coef * kl_budget)

            ## Other variants/hyperparamter(s)
            target_actor=True,
            clip_adj=True,
            clip_grad=True,
            use_target_grad=True,
        )
    )
    return config