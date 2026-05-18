from agents.fql import FQLAgent
from agents.cgql import CGQLAgent
from agents.qam import QAMAgent
from agents.trqam import TRQAMAgent
from agents.dsrl import DSRLAgent
from agents.ifql import IFQLAgent

agents = dict(
    ifql=IFQLAgent,
    fql=FQLAgent,
    dsrl=DSRLAgent,
    qam=QAMAgent,
    trqam=TRQAMAgent,
    cgql=CGQLAgent,
)
