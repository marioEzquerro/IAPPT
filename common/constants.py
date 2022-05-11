LABELS = ['WRIST_Y','WRIST_X',
'INDEX_FINGER_PIP_Y','INDEX_FINGER_PIP_X',
'INDEX_FINGER_TIP_Y','INDEX_FINGER_TIP_X',
'MIDDLE_FINGER_PIP_Y','MIDDLE_FINGER_PIP_X',
'MIDDLE_FINGER_TIP_Y','MIDDLE_FINGER_TIP_X',
'RING_FINGER_PIP_Y','RING_FINGER_PIP_X',
'RING_FINGER_TIP_Y','RING_FINGER_TIP_X',
'PINKY_FINGER_PIP_Y','PINKY_FINGER_PIP_X',
'PINKY_FINGER_TIP_Y','PINKY_FINGER_TIP_X',
]
GESTURES = ['Piedra','Papel','Tijera']
TRAINING_FOLDERS = ['.\\data\\train\\paper', '.\\data\\train\\scissors', '.\\data\\train\\rock']
TESTIING_FOLDERS = ['.\\data\\test\\paper', '.\\data\\test\\scissors', '.\\data\\test\\rock']
TRAINING_CSV = '.\\data\\training_data.csv'
EVALUATION_CSV = '.\\data\\eval_data.csv'
MAX_SCORE = 5
ROCK = 0
PAPER = 1
SCISSORS = 2