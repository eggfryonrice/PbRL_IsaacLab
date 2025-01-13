import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# Camera parameters
camera_target = np.array([50, 0, 40], dtype=np.float32)
camera_eye = np.array([50, -400, 40], dtype=np.float32)
theta1 = 0  # Vertical rotation (up-down)
theta2 = 0  # Horizontal rotation (left-right)
distance = 400

theta1_min = np.radians(-5)
theta1_max = np.radians(85)
rot_speed = 0.1


def update_camera_eye():
    global camera_eye
    x = distance * np.cos(theta1) * np.sin(theta2)
    y = -distance * np.cos(theta1) * np.cos(theta2)
    z = distance * np.sin(theta1)
    camera_eye = camera_target + np.array([x, y, z], dtype=np.float32)


# size multiplier
zoom = 100

# interval multiplier - gets (slower) times slower
slower = 1

# primitive size
sphere_radius = 3
cuboid_width = 4

# UI page size
UI_width, UI_height = 3500, 2000
button_height = 300  # height for buttons
gap = 20  # gap between buttons
sub_screen_width = 400

# light and material properties
light_ambient = [0.2, 0.2, 0.2, 1.0]
light_diffuse = [0.8, 0.8, 0.8, 1.0]
light_specular = [1.0, 1.0, 1.0, 1.0]
material_specular = [0.1, 0.1, 0.1, 1.0]
material_shininess = [10.0]

# Type hints
sphereInput = tuple[np.ndarray, tuple[float, float, float]]
linkInput = tuple[np.ndarray, float, np.ndarray, tuple[float, float, float]]
capsuleInput = tuple[np.ndarray, np.ndarray, float, tuple[float, float, float]]
sceneInput = tuple[list[capsuleInput], list[sphereInput], list[linkInput]]


def draw_chessboard():
    numGrid = 10
    blockSize = 50
    height = -sphere_radius
    halfSize = (numGrid * blockSize) / 2

    for i in range(numGrid):
        for j in range(numGrid):
            x = -halfSize + i * blockSize
            y = -halfSize + j * blockSize

            if (i + j) % 2 == 0:
                glColor3f(0.9, 0.9, 0.9)
            else:
                glColor3f(0.1, 0.1, 0.1)

            glBegin(GL_QUADS)
            glVertex3f(x, y, height)
            glVertex3f(x + blockSize, y, height)
            glVertex3f(x + blockSize, y + blockSize, height)
            glVertex3f(x, y + blockSize, height)
            glEnd()


def draw_sphere(sphere: sphereInput):
    position, color = sphere
    position = position * zoom
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor3f(color[0], color[1], color[2])
    quadric = gluNewQuadric()
    gluSphere(quadric, sphere_radius, 20, 20)
    gluDeleteQuadric(quadric)
    glPopMatrix()


def draw_cuboid(link: linkInput):
    start_pos, length, quat, color = link
    start_pos = start_pos * zoom
    length = length * zoom

    glPushMatrix()

    # Translate to the starting position
    glTranslatef(start_pos[0], start_pos[1], start_pos[2])

    # Apply the rotation from the quaternion
    angle = 2 * math.degrees(math.acos(quat[0]))
    axis = quat[1:]
    if np.linalg.norm(axis) > 1e-6:
        glRotatef(angle, axis[0], axis[1], axis[2])

    # Set color
    glColor3f(color[0], color[1], color[2])
    # Scale to form the cuboid
    glScalef(cuboid_width, cuboid_width, length)

    # Draw the cuboid
    vertices = [
        [0.5, -0.5, sphere_radius / length],
        [0.5, 0.5, sphere_radius / length],
        [-0.5, 0.5, sphere_radius / length],
        [-0.5, -0.5, sphere_radius / length],
        [0.5, -0.5, 1.0 - sphere_radius / length],
        [0.5, 0.5, 1.0 - sphere_radius / length],
        [-0.5, -0.5, 1.0 - sphere_radius / length],
        [-0.5, 0.5, 1.0 - sphere_radius / length],
    ]

    faces = [
        [0, 1, 2, 3],  # Front
        [5, 4, 6, 7],  # Back
        [3, 2, 7, 6],  # Left
        [1, 0, 4, 5],  # Right
        [2, 1, 5, 7],  # Top
        [0, 3, 6, 4],  # Bottom
    ]

    normals = [
        [0, 0, -1],
        [0, 0, 1],
        [-1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
    ]

    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glNormal3fv(normals[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    glPopMatrix()


def draw_capsule(capsule: capsuleInput):
    start_pos, end_pos, radius, color = capsule
    start_pos = start_pos * zoom
    end_pos = end_pos * zoom
    radius = radius * zoom

    glPushMatrix()
    glColor3f(color[0], color[1], color[2])

    direction = end_pos - start_pos
    length = np.linalg.norm(direction)
    if length < 0:
        return

    # Translate
    glTranslatef(start_pos[0], start_pos[1], start_pos[2])

    if length > 1e-6:
        direction_normalized = direction / length
        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_axis = np.cross(z_axis, direction_normalized)
        rotation_angle = math.degrees(math.acos(np.dot(z_axis, direction_normalized)))

        if np.linalg.norm(rotation_axis) > 1e-6:
            glRotatef(
                rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2]
            )

    quadric = gluNewQuadric()
    if length > 1e-6:
        gluCylinder(quadric, radius, radius, length, 20, 20)

    glPushMatrix()
    gluSphere(quadric, radius, 20, 20)  # Bottom hemisphere
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0, 0, length)
    gluSphere(quadric, radius, 20, 20)  # Top hemisphere
    glPopMatrix()

    gluDeleteQuadric(quadric)
    glPopMatrix()


def render_scene(input: sceneInput):
    capsules, spheres, links = input

    draw_chessboard()
    for sphere in spheres:
        draw_sphere(sphere)
    for link in links:
        draw_cuboid(link)
    for capsule in capsules:
        draw_capsule(capsule)


def event_handler():
    """
    Same event-handling logic as before:
    returns highlight booleans, running state, choice, best, etc.
    """
    global theta1, theta2
    mouse_pos = pygame.mouse.get_pos()
    mouse_pos = (mouse_pos[0], UI_height - mouse_pos[1])
    highlight_left_best_button = False
    highlight_right_best_button = False
    highlight_left_scene = False
    highlight_right_scene = False
    highlight_skip_button = False
    highlight_no_pref_button = False
    running = True
    choice = None
    best = None

    if (
        gap <= mouse_pos[0]
        and mouse_pos[0]
        <= gap + ((UI_width - gap - sub_screen_width) // 2) - (gap // 2)
        and UI_height - button_height - gap <= mouse_pos[1]
        and mouse_pos[1] <= UI_height - gap
    ):
        highlight_left_best_button = True
    elif (
        (UI_width // 2) + (gap // 2) <= mouse_pos[0]
        and mouse_pos[0] <= UI_width - sub_screen_width - 2 * gap
        and UI_height - button_height - gap <= mouse_pos[1]
        and mouse_pos[1] <= UI_height - gap
    ):
        highlight_right_best_button = True
    elif (
        gap <= mouse_pos[0]
        and mouse_pos[0] <= gap + (UI_width // 2) - (gap // 2)
        and button_height + gap + gap <= mouse_pos[1]
        and mouse_pos[1] <= UI_height - gap - button_height - gap
    ):
        highlight_left_scene = True
    elif (
        (UI_width // 2) + (gap // 2) <= mouse_pos[0]
        and mouse_pos[0] <= UI_width - gap
        and button_height + gap + gap <= mouse_pos[1]
        and mouse_pos[1] <= UI_height - gap - button_height - gap
    ):
        highlight_right_scene = True
    elif (
        gap <= mouse_pos[0]
        and mouse_pos[0] <= (UI_width / 2) - (gap / 2)
        and gap <= mouse_pos[1]
        and mouse_pos[1] <= button_height + gap
    ):
        highlight_skip_button = True
    elif (
        (UI_width / 2) + (gap / 2) <= mouse_pos[0]
        and mouse_pos[0] <= UI_width - gap
        and gap <= mouse_pos[1]
        and mouse_pos[1] <= button_height + gap
    ):
        highlight_no_pref_button = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            choice = None
        elif event.type == pygame.MOUSEBUTTONUP:
            running = False
            if highlight_left_scene:
                choice = 0
            elif highlight_right_scene:
                choice = 1
            elif highlight_left_best_button:
                choice = 0
                best = 0
            elif highlight_right_best_button:
                choice = 1
                best = 1
            elif highlight_skip_button:
                choice = None
            elif highlight_no_pref_button:
                choice = -1
            else:
                running = True

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        theta1 += rot_speed
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        theta1 -= rot_speed
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        theta2 -= rot_speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        theta2 += rot_speed

    theta1 = max(theta1_min, min(theta1_max, theta1))
    update_camera_eye()

    return (
        highlight_left_best_button,
        highlight_right_best_button,
        highlight_left_scene,
        highlight_right_scene,
        highlight_skip_button,
        highlight_no_pref_button,
        running,
        choice,
        best,
    )


class PreferenceLabeler:
    """
    We do pygame.init() ONCE (in __init__).
    Then in label_preference, we set up the big window,
    run the loop, and at the end we MINIMIZE the window
    so it doesn't hog CPU or remain visible.
    """

    def __init__(self):
        # Initialize Pygame globally (once for the entire script)
        pygame.init()
        pygame.display.set_mode((UI_width, UI_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Preference Labeling")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess)

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)

    def label_preference(self, frames1, frames2, best_frames, interval=1 / 60):
        clock = pygame.time.Clock()

        frame_index = 0
        running = True
        choice = None
        best = None

        while running:
            (
                highlight_left_best_button,
                highlight_right_best_button,
                highlight_left_scene,
                highlight_right_scene,
                highlight_skip_button,
                highlight_no_pref_button,
                running,
                choice,
                best,
            ) = event_handler()

            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glColor3f(0.5, 0.5, 1)

            glViewport(0, 0, UI_width, UI_height)

            # ----- EXACT SAME UI DRAWING / BUTTON CODE -----
            button_surface = pygame.Surface((UI_width, UI_height), pygame.SRCALPHA)
            button_surface.fill((0, 0, 0, 0))

            skip_rect = pygame.Rect(
                gap,
                UI_height - button_height - gap,
                (UI_width // 2) - (gap // 2) - gap,
                button_height,
            )
            button_color = (
                (255, 255, 200, 255) if highlight_skip_button else (200, 200, 200, 255)
            )
            pygame.draw.rect(button_surface, button_color, skip_rect)
            skip_text = self.font.render("Skip", True, (0, 0, 0))
            text_rect = skip_text.get_rect(center=skip_rect.center)
            button_surface.blit(skip_text, text_rect)

            no_pref_rect = pygame.Rect(
                (UI_width // 2) + (gap // 2),
                UI_height - button_height - gap,
                (UI_width // 2) - (gap // 2) - gap,
                button_height,
            )
            button_color = (
                (255, 255, 200, 255)
                if highlight_no_pref_button
                else (200, 200, 200, 255)
            )
            pygame.draw.rect(button_surface, button_color, no_pref_rect)
            no_pref_text = self.font.render("Equally Preferable", True, (0, 0, 0))
            text_rect = no_pref_text.get_rect(center=no_pref_rect.center)
            button_surface.blit(no_pref_text, text_rect)

            left_best_rect = pygame.Rect(
                gap,
                gap,
                ((UI_width - sub_screen_width - gap) // 2) - (gap // 2) - gap,
                button_height,
            )
            button_color = (
                (255, 255, 200, 255)
                if highlight_left_best_button
                else (200, 200, 200, 255)
            )
            pygame.draw.rect(button_surface, button_color, left_best_rect)
            left_best_text = self.font.render("query 0 is best so far", True, (0, 0, 0))
            text_rect = left_best_text.get_rect(center=left_best_rect.center)
            button_surface.blit(left_best_text, text_rect)

            right_best_rect = pygame.Rect(
                ((UI_width - sub_screen_width - gap) // 2) + (gap // 2),
                gap,
                ((UI_width - sub_screen_width - gap) // 2) - (gap // 2) - gap,
                button_height,
            )
            button_color = (
                (255, 255, 200, 255)
                if highlight_right_best_button
                else (200, 200, 200, 255)
            )
            pygame.draw.rect(button_surface, button_color, right_best_rect)
            right_best_text = self.font.render(
                "query 1 is best so far", True, (0, 0, 0)
            )
            text_rect = right_best_text.get_rect(center=right_best_rect.center)
            button_surface.blit(right_best_text, text_rect)

            # Convert the surface to texture
            button_texture_data = pygame.image.tostring(button_surface, "RGBA", True)

            button_texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, button_texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                button_surface.get_width(),
                button_surface.get_height(),
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                button_texture_data,
            )

            # Orthographic pass for UI
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, UI_width, 0, UI_height, -1, 1)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)

            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, button_texture_id)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 0)
            glVertex2f(0, 0)
            glTexCoord2f(1, 0)
            glVertex2f(UI_width, 0)
            glTexCoord2f(1, 1)
            glVertex2f(UI_width, UI_height)
            glTexCoord2f(0, 1)
            glVertex2f(0, UI_height)
            glEnd()

            glDisable(GL_TEXTURE_2D)

            # Restore
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)

            glDeleteTextures([button_texture_id])

            # ----- Render left scene -----
            scene_width = (UI_width // 2) - (gap // 2) - gap
            if highlight_left_scene:
                glClearColor(0.1, 0.1, 0.1, 1.0)
            else:
                glClearColor(0.0, 0.0, 0.0, 1.0)

            glViewport(
                gap,
                button_height + gap + gap,
                scene_width,
                UI_height - (2 * button_height + 4 * gap),
            )
            glScissor(
                gap,
                button_height + gap + gap,
                scene_width,
                UI_height - (2 * button_height + 4 * gap),
            )
            glEnable(GL_SCISSOR_TEST)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_SCISSOR_TEST)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(
                45,
                scene_width / (UI_height - (2 * button_height + 4 * gap)),
                0.1,
                5000.0,
            )
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(
                camera_eye[0],
                camera_eye[1],
                camera_eye[2],
                camera_target[0],
                camera_target[1],
                camera_target[2],
                0,
                0,
                1,
            )
            render_scene(frames1[frame_index])

            # ----- Render right scene -----
            if highlight_right_scene:
                glClearColor(0.1, 0.1, 0.1, 1.0)
            else:
                glClearColor(0.0, 0.0, 0.0, 1.0)

            glViewport(
                (UI_width // 2) + (gap // 2),
                button_height + gap + gap,
                scene_width,
                UI_height - (2 * button_height + 4 * gap),
            )
            glScissor(
                (UI_width // 2) + (gap // 2),
                button_height + gap + gap,
                scene_width,
                UI_height - (2 * button_height + 4 * gap),
            )
            glEnable(GL_SCISSOR_TEST)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDisable(GL_SCISSOR_TEST)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(
                45,
                scene_width / (UI_height - (2 * button_height + 4 * gap)),
                0.1,
                5000.0,
            )
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(
                camera_eye[0],
                camera_eye[1],
                camera_eye[2],
                camera_target[0],
                camera_target[1],
                camera_target[2],
                0,
                0,
                1,
            )
            render_scene(frames2[frame_index])

            # ----- Render best_frames sub-scene -----
            if best_frames is not None:
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glViewport(
                    UI_width - sub_screen_width - gap,
                    UI_height - button_height - gap,
                    sub_screen_width,
                    button_height,
                )
                glScissor(
                    UI_width - sub_screen_width - gap,
                    UI_height - button_height - gap,
                    sub_screen_width,
                    button_height,
                )
                glEnable(GL_SCISSOR_TEST)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glDisable(GL_SCISSOR_TEST)

                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45, sub_screen_width / button_height, 0.1, 5000.0)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                gluLookAt(
                    camera_eye[0],
                    camera_eye[1],
                    camera_eye[2],
                    camera_target[0],
                    camera_target[1],
                    camera_target[2],
                    0,
                    0,
                    1,
                )
                render_scene(best_frames[frame_index])

            pygame.display.flip()

            frame_index += 1
            if frame_index >= len(frames1):
                clock.tick(1 / 0.5)
                frame_index = 0

            clock.tick(1 / (slower * interval))

        # Instead of pygame.quit(), just MINIMIZE the window:
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        pygame.display.flip()

        # Return the user’s choice
        print(f"User choice: {choice}    ", end="\r", flush=True)
        return choice, best


# ------------------- Example usage -------------------
if __name__ == "__main__":

    def create_dummy_frame(frame_number):
        capsules = [
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 2.0, 1.0]), 0.2, (1, 1, 0.5))
        ]
        jointsPositions = [
            (np.array([0.0, 0.0, 0.0]), (1, 0.5, 0.5)),
            (np.array([50.0, 0.0, 0.0]), (0.5, 1, 0.5)),
        ]
        links = [
            (
                np.array([0.0, 0.0, 0.0]),
                1,
                np.array([1.0, 0.0, 0.0, 0.0]),
                (0.5, 0.5, 1),
            )
        ]
        return (capsules, jointsPositions, links)

    frames1 = [create_dummy_frame(i) for i in range(60)]
    frames2 = [create_dummy_frame(i) for i in range(60)]

    # 1) Create the labeler (initializes Pygame once)
    labeler = PreferenceLabeler()

    # 2) Call label_preference
    choice, best = labeler.label_preference(frames1, frames2, frames1)
    print("User choice:", choice)
