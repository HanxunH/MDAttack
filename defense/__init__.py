from . import RST, UAT, TRADES, MART, MMA, BAT, ADVInterp, FeaScatter, Sense
from . import JARN_AT, Dynamic, AWP, Overfitting, ATHE, PreTrain, SAT
from . import RobustWRN

defence_options = {
    "RST": RST.DefenseRST,
    "UAT": UAT.DefenseUAT,
    "TRADES": TRADES.DefenseTRADES,
    "MART": MART.DefenseMART,
    "MMA": MMA.DefenseMMA,
    "BAT": BAT.DefenseBAT,
    "ADVInterp": ADVInterp.DefenseADVInterp,
    "FeaScatter": FeaScatter.DefenseFeaScatter,
    "Sense": Sense.DefenseSense,
    "JARN_AT": JARN_AT.DefenseJARN_AT,
    "Dynamic": Dynamic.DefenseDynamic,
    "AWP": AWP.DefenseAWP,
    "Overfitting": Overfitting.DefenseOverfitting,
    "ATHE": ATHE.DefenseATHE,
    "PreTrain": PreTrain.DefensePreTrain,
    "SAT": SAT.DefenseSAT,
    "RobustWRN": RobustWRN.DefenseRobustWRN,
}
