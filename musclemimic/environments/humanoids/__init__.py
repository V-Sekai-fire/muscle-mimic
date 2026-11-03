from .myofullbody import MyoFullBody, MjxMyoFullBody
from .ms_human_700 import MsHuman700MyoFullBody, MjxMsHuman700MyoFullBody
from .bimanual import MyoBimanualArm, MjxMyoBimanualArm


# register muscle environments
MyoBimanualArm.register()
MjxMyoBimanualArm.register()
MyoFullBody.register()
MjxMyoFullBody.register()
MsHuman700MyoFullBody.register()
MjxMsHuman700MyoFullBody.register()
