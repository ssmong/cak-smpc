"""
Random Controller for Training Data Generation
Outputs random control torques within specified bounds.
"""

import numpy as np
from typing import Optional

from Basilisk.architecture import messaging
from Basilisk.architecture import sysModel


class Random(sysModel.SysModel):
    """
    Random controller that generates random torques for data generation.
    """
    
    def __init__(self, torque_max: float, seed: Optional[int] = None, **kwargs):
        super().__init__()
        
        self.ModelTag = "RandomController"
        self.torque_max = torque_max
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Create output message
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        
        # Create input messages (unused but needed for compatibility)
        self.attRefInMsg = messaging.AttRefMsg()
        
        # Statistics tracking (for API compatibility)
        self._wall_times = []
        self._total_times = []
    
    def Reset(self, CurrentSimNanos):
        self.rng = np.random.RandomState(self.seed)
        return
    
    def UpdateState(self, currentSimNanos):
        # Generate random torque vector
        torque = self.rng.uniform(-self.torque_max, self.torque_max, size=3)
        
        # Write output message
        cmdTorqueMsg = messaging.CmdTorqueBodyMsgPayload()
        cmdTorqueMsg.torqueRequestBody = torque
        self.cmdTorqueOutMsg.write(cmdTorqueMsg, self.moduleID, currentSimNanos)
        
        return
    
    def get_wall_time_stats(self):
        """Dummy implementation"""
        return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}
    
    def get_total_time_stats(self):
        """Dummy implementation"""
        return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0, 'count': 0}
