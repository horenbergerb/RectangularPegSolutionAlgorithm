import tkinter as tk
import numpy as np

#The whole point here is basically to let you draw curves that you can plug into the program
#I'm gonna make it so this outputs a file and you can pipe that to the other program somehow

#distance between mouse locations before we add another point to our array
res = 50.0

points = []
cur_len = 0
is_pressed = 0

window = tk.Tk()

def press(event=None):
    global is_pressed
    is_pressed = 1
    
def pressed_motion(event=None):
    global points
    global cur_len
    global is_pressed
    if is_pressed == 0:
        return
    if len(points) == 0:
        points.append(np.array([event.x, event.y]))
    else:
        cur = np.array([event.x, event.y])
        if np.linalg.norm(points[cur_len]-cur) >= res:
            points.append(cur)
            cur_len += 1
            canvas.create_oval(event.x-1, event.y-1, event.x+1, event.y+1)
            
def release(event=None):
    global is_pressed
    global points
    print(points)
    is_pressed = 0
    #save our shape
    points = np.array(points)
    np.savetxt("customshape.txt", points)
    #quit program
    quit()

canvas = tk.Canvas(window, width=1000, height=1000)
canvas.pack()
canvas.bind("<Button-1>", press)
canvas.bind("<B1-Motion>", pressed_motion)
canvas.bind("<ButtonRelease-1>", release)

tk.mainloop()
