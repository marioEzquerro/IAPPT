# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requi_RED by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe solution drawing styles."""

from doctest import ELLIPSIS_MARKER
from typing import Mapping, Tuple

from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark

_RADIUS = 4
_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (110, 110, 110)
_PURPLE = (128, 64, 128)
# _PEACH = (180, 229, 255)
_PEACH = (255, 255, 255)
_WHITE = (224, 224, 224)
_ORANGE = (0, 140, 255)

# Tamaño de lso nodos de las landmarks
_THICKNESS_WRIST_MCP = 3
_THICKNESS_FINGER = 2
_THICKNESS_DOT = -1

_PALM_LANMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                  HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP,
                  HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP)
_THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,
                    HandLandmark.THUMB_TIP)
_INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP,
                           HandLandmark.INDEX_FINGER_DIP,
                           HandLandmark.INDEX_FINGER_TIP)
_MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP,
                            HandLandmark.MIDDLE_FINGER_DIP,
                            HandLandmark.MIDDLE_FINGER_TIP)
_RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP,
                          HandLandmark.RING_FINGER_DIP,
                          HandLandmark.RING_FINGER_TIP)
_PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP,
                           HandLandmark.PINKY_TIP)

# DIFERENTES ESTILOS POR CADA GESTO
# Landmarks
_HAND_LANDMARK_STYLE_ROCK = {
    _PALM_LANMARKS:
        DrawingSpec(
            color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS:
        DrawingSpec(
            color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}
_HAND_LANDMARK_STYLE_PAPER = {
    _PALM_LANMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}
_HAND_LANDMARK_STYLE_SCISSORS = {
    _PALM_LANMARKS:
        DrawingSpec(
            color=_ORANGE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS:
        DrawingSpec(
            color=_ORANGE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=_ORANGE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=_ORANGE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=_ORANGE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=_ORANGE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

# Conexiones de la mano
_HAND_CONNECTION_STYLE_ROCK = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_GRAY, thickness=_THICKNESS_FINGER)
}

_HAND_CONNECTION_STYLE_PAPER = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER)
}

_HAND_CONNECTION_STYLE_SCISSORS = {
    hands_connections.HAND_PALM_CONNECTIONS:
        DrawingSpec(color=_ORANGE, thickness=_THICKNESS_WRIST_MCP),
    hands_connections.HAND_THUMB_CONNECTIONS:
        DrawingSpec(color=_ORANGE, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=_ORANGE, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=_ORANGE, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_RING_FINGER_CONNECTIONS:
        DrawingSpec(color=_ORANGE, thickness=_THICKNESS_FINGER),
    hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=_ORANGE, thickness=_THICKNESS_FINGER)
}


def get_hand_landmarks_style(accion) -> Mapping[int, DrawingSpec]:
    """Returns the default hand landmarks drawing style.

    Returns:
        A mapping from each hand landmark to its default drawing spec.
    """
    hand_landmark_style = {}
    global hand_landmarks

    if (accion == 0):
        hand_landmarks = _HAND_LANDMARK_STYLE_ROCK
    elif (accion == 1):
        hand_landmarks = _HAND_LANDMARK_STYLE_PAPER
    else:
        hand_landmarks = _HAND_LANDMARK_STYLE_SCISSORS


    for k, v in hand_landmarks.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style


def get_hand_connections_style(accion) -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default hand connections drawing style.

    Returns:
        A mapping from each hand connection to its default drawing spec.
    """
    hand_connection_style = {}
    global hand_connections

    if (accion == 0):
        hand_connections = _HAND_CONNECTION_STYLE_ROCK
    elif (accion == 1):
        hand_connections = _HAND_CONNECTION_STYLE_PAPER
    else:
        hand_connections = _HAND_CONNECTION_STYLE_SCISSORS

    for k, v in hand_connections.items():
        for connection in k:
            hand_connection_style[connection] = v

    return hand_connection_style
