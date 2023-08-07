import win32api, win32con, time
def click(interval):
    print("start")
    t0 = time.time()
    x, y = 512, 512
    while  time.time()-t0<interval:
        win32api.SetCursorPos((x,y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
        time.sleep(300)
click(60*60*2)

