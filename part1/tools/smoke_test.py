import time
import agibot_gdk

assert agibot_gdk.gdk_init() == agibot_gdk.GDKRes.kSuccess

try:
    camera = agibot_gdk.Camera()
    time.sleep(3)

    img = camera.get_latest_image(agibot_gdk.CameraType.kHeadColor, 1000.0)

    print("Image:")
    print("width:", img.width)
    print("height:", img.height)
    print("encoding:", img.encoding)
    print("color_format:", img.color_format)
    print("data type:", type(img.data))
    print("shape:", getattr(img.data, "shape", None))

    camera.close_camera()

finally:
    agibot_gdk.gdk_release()