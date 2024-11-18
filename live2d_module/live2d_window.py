import random

import live2d.utils.log as log
import live2d.v3 as live2d
from PySide2.QtCore import QTimer
from PySide2.QtGui import QSurfaceFormat
from PySide2.QtWidgets import QOpenGLWidget

from live2d_module.lip_sync import WavHandler

# 初始化日志
live2d.setLogEnable(False)

count = 0

from PySide2.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide2.QtCore import Qt


class OpenGLLive2DWidget(QOpenGLWidget):
    def __init__(self, parent=None, live2d_model=''):
        super(OpenGLLive2DWidget, self).__init__(parent)
        self.model_path = live2d_model
        self.model = None
        self.dx = 0.0
        self.dy = 0.05
        self.scale = 1.1
        self.audioPlayed = False
        self.wavHandler = WavHandler()
        self.lipSyncN = 4

        # 配置 OpenGL 上下文
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        self.setFormat(fmt)

    def initializeGL(self):
        # 初始化 OpenGL 环境和 Live2D
        live2d.init()
        live2d.glewInit()
        live2d.setGLProperties()

        # 加载模型
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(self.model_path)
        self.model.Resize(900, 900)
        self.model.SetAutoBlinkEnable(True)
        self.model.SetAutoBreathEnable(True)

        # 定时刷新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()

    def paintGL(self):
        if self.model:
            self.model.Update()
            if self.wavHandler.Update():
                openY = self.wavHandler.GetRms() * self.lipSyncN
                self.model.AddParameterValue("ParamMouthOpenY", openY if openY > 0.15 else 0)

            if not self.audioPlayed:
                self.audioPlayed = True

            # 渲染模型
            self.model.SetOffset(self.dx, self.dy)
            self.model.SetScale(self.scale)
            live2d.clearBuffer(1.0, 1.0, 1.0, 0.0)
            self.model.Draw()

    def speech(self, audio_path):
        log.Info("start lipSync")
        self.wavHandler.Start(audio_path)

    def resizeGL(self, w, h):
        # 调整视口大小
        # gl.glViewport(0, 0, w, h)
        self.model.Resize(w, h)

    def updateGL(self):
        self.update()

    def idle(self, i=0, _random=True):
        if _random:
            i = random.randint(0, 3)
        self.model.StartMotion('Idle', i, live2d.MotionPriority.IDLE.value, self.on_start_motion_callback)

    def tap(self, i=0, _random=True):
        motions = ['Flick', 'FlickDown', 'FlickUp', 'Tap', 'Tap@Body', 'Flick@Body']
        count_motion = [1, 1, 1, 2, 1, 1]
        if _random:
            i = random.randint(0, sum(count_motion))

        for motion, j in zip(motions, count_motion):
            for k in range(0, j):
                if i == k:
                    self.model.StartMotion(motion, k, live2d.MotionPriority.FORCE.value, self.on_start_motion_callback)
                    return
                i -= 1

    def on_start_motion_callback(self, group: str, no: int):
        print(f'group:{group} no:{no}')

    def on_finish_motion_callback(self):
        log.Info("motion finished")

    def keyPressEvent(self, event):
        print(event)
        if event.key() == Qt.Key_Left:
            self.dx -= 0.1
        elif event.key() == Qt.Key_Right:
            self.dx += 0.1
        elif event.key() == Qt.Key_Up:
            self.dy += 0.1
        elif event.key() == Qt.Key_Down:
            self.dy -= 0.1
        elif event.key() == Qt.Key_I:
            self.scale += 0.01
        elif event.key() == Qt.Key_U:
            self.scale -= 0.01
        elif event.key() == Qt.Key_Space:
            global count
            motions = ['Idle', 'Flick', 'FlickDown', 'FlickUp', 'Tap', 'Tap@Body', 'Flick@Body']
            count_motion = [3, 1, 1, 1, 2, 1, 1]
            tmp = 0
            for i, j in zip(motions, count_motion):
                flag = False
                for k in range(0, j):
                    if tmp == count:
                        print('======================')
                        print(i, k)
                        self.model.StartMotion(i, k, live2d.MotionPriority.FORCE.value, self.on_start_motion_callback)
                        flag = True
                        break
                    tmp += 1
                if flag:
                    break
            count += 1
            count %= 10

    def mousePressEvent(self, event):
        if self.model:
            x, y = event.x(), event.y()
            self.tap(_random=True)
            self.model.Touch(x, y, self.on_start_motion_callback, self.on_finish_motion_callback)

    def mouseMoveEvent(self, event):
        if self.model:
            self.model.Drag(event.x(), event.y())


class Live2DWidget(QWidget):
    def __init__(self, parent=None, live2d_model=''):
        super(Live2DWidget, self).__init__(parent)

        # 设置尺寸
        self.setGeometry(0, 0, 900, 900)

        # 创建布局
        self.layout = QVBoxLayout(self)

        # 创建OpenGL widget
        self.opengl_widget = OpenGLLive2DWidget(self, live2d_model=live2d_model)

        # 添加到布局
        self.layout.addWidget(self.opengl_widget)

        # 设置布局
        self.setLayout(self.layout)

        # 设置标签
        self.speech_bubble = QLabel(self)
        self.speech_bubble.setStyleSheet("""
                    QLabel {
                        background-color: rgba(255, 255, 255, 0.9);
                        border: 5px solid #333333;
                        border-radius: 20px;
                        padding: 20px;
                        color: #000000;
                        font-size: 30px;
                    }
                """)
        self.speech_bubble.setWordWrap(True)

        self.speech_bubble.raise_()
        self.speech_bubble.setFocusPolicy(Qt.NoFocus)
        self.speech_bubble.hide()

        # 创建定时器
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)  # 只触发一次
        self.timer.timeout.connect(self.hide_speech_bubble)  # 连接超时信号

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.speech_bubble.isHidden():
            x = max(0, self.width() // 2 - 450)
            y = max(0, self.height() // 2 - 450)
            self.speech_bubble.move(x, y)

    # 转发鼠标事件到OpenGL widget
    def mousePressEvent(self, event):
        self.opengl_widget.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.opengl_widget.mouseMoveEvent(event)

    def keyPressEvent(self, event):
        self.opengl_widget.keyPressEvent(event)

    def hide_speech_bubble(self):
        self.speech_bubble.hide()  # 隐藏气泡

    def speech(self, text, audio_path):
        self.opengl_widget.idle(_random=True)

        self.speech_bubble.setText(text)
        self.speech_bubble.adjustSize()
        self.speech_bubble.show()  # 显示气泡

        x = max(0, self.width() // 2 - 450)
        y = max(0, self.height() // 2 - 450)

        self.speech_bubble.move(x, y)

        self.opengl_widget.speech(audio_path)

        # 启动定时器，6秒后自动隐藏
        self.timer.start(6000)
