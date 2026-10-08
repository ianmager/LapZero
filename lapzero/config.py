#window
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
FPS = 60
WINDOW_TITLE = "LapZero — Track Editor"

#colors
BACKGROUND_COLOR = (24, 26, 32)
GRID_COLOR = (34, 37, 46)
BOUNDARY_COLOR = (235, 235, 240)
IN_PROGRESS_COLOR = (120, 200, 255)
START_COLOR = (80, 220, 120)
FINISH_COLOR = (230, 90, 90)
TEXT_COLOR = (210, 210, 215)
HUD_BG_COLOR = (16, 17, 21)
STATUS_COLOR = (255, 210, 90)

#editor tuning
POINT_RADIUS = 9
BOUNDARY_LINE_WIDTH = 3
GRID_SPACING = 40
MIN_STROKE_POINT_DISTANCE = 6

#persistence
TRACKS_DIR = "tracks"
DEFAULT_TRACK_NAME = "track"

#racer physics
RACER_RADIUS = 10
RACER_COLOR = (255, 200, 60)
RACER_MAX_SPEED = 260.0        # px/sec
RACER_MIN_SPEED = -100.0       # allows slow reverse
RACER_ACCELERATION = 220.0     # px/sec^2 while throttling
RACER_BRAKE_DECEL = 260.0      # px/sec^2 while braking/reversing
RACER_FRICTION_DECEL = 140.0   # px/sec^2 natural coast-down, no input
RACER_MAX_TURN_RATE = 3.2      # rad/sec, at zero speed (sharpest turn)
RACER_MIN_TURN_RATE = 0.9      # rad/sec, at max speed (widest turn)

#raycast sensing
RAY_COUNT = 7
RAY_SPREAD_DEGREES = 180.0     # total field of view, centered on heading
RAY_MAX_DISTANCE = 400.0
RAY_COLOR = (90, 160, 255)
RAY_HIT_COLOR = (255, 120, 120)

#neural net
NN_HIDDEN_SIZE = 6
NN_STEER_DEADZONE = 0.2
NN_THROTTLE_DEADZONE = 0.2
RACER_AI_COLOR = (120, 210, 255)
RACER_DEAD_COLOR = (110, 110, 120)
RACER_FINISHED_COLOR = (80, 220, 120)
AI_CRUISE_SPEED = 110.0        # px/sec cap while wandering; keeps turns tight enough to avoid walls

#evolution
FINISH_RADIUS = 42.0
EXPLORE_CELL = 36              # new ground scored in cells of this size
GA_POPULATION = 40
GA_ELITE = 4
GA_TOURNAMENT = 4
GA_MUTATION_RATE = 0.15
GA_MUTATION_SCALE = 0.22
GA_FINE_MUTATION_RATE = 0.12   # small tweaks of the lap record, once it exists
GA_FINE_MUTATION_SCALE = 0.07
GA_GENERATION_SECONDS = 18.0
GA_STALL_SECONDS = 5.0         # die if no new ground is covered in this long
GA_STEPS_PER_FRAME = 1
GA_FAST_STEPS = 4
