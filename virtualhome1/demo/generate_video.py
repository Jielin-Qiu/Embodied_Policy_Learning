# Generate video for a program. Make sure you have the executable open
import sys
sys.path.append('../simulation/')
from unity_simulator.comm_unity import UnityCommunication
script = ['[Walk] <sofa> (1)', '[Sit] <sofa> (1)'] # Add here your script
print('Starting Unity...')
comm = UnityCommunication()
print('Starting scene...')
comm.reset()
print('Generating video...')
# for i in script:
#     comm.reset()
#     comm.render_script([i], camera_mode = 'FIRST_PERSON', file_name_prefix = f'{i}',find_solution = True,capture_screenshot=True, processing_time_limit=60)
#     print('Generated, find video in simulation/unity_simulator/output/')
#     s, cc = comm.camera_count()
s, cc = comm.camera_count()
for script_instruction in script:
    comm.render_script([script_instruction])
    # Here you can get an observation, for instance
