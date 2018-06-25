#!/bin/python3
##!/cygdrive/c/Python35-32/python -u

# flip.py
# image flipper -- checks alignment of two images

# Daniel Scharstein
# v1.0 6/14/2011 - first working version
# v1.1 6/20/2011 - faster showing of imgs and diff, slightly better centering
# v1.2 6/30/2011 - shows color under cursor
# v1.3 7/15/2011 - major clean up, used tuples everywhere, changed representation of crop area
# v1.4 7/7/2016 - ported to python3, Pillow

# TODO:
# add zoom out feature!
# (need to figure out how to clear image area out of bounds)


import os, sys, glob
import tkinter as tk
from PIL import Image, ImageTk, ImageChops

if len(sys.argv) != 3:
    print("usage: %s im1 im2" % sys.argv[0])
    sys.exit()

instructions = """\
Drag: move     Tab: blink    c: center    z/x: zoom in/out     Esc,q: quit
Shift: show diff     Shift-drag, arrows: move relative"""


class flipper:

    def __init__(self, imn1, imn2):
        self.root = tk.Tk()
        #self.root.configure(bg = "blue")
        ww = min(700, self.root.winfo_screenwidth() - 100)
        wh = min(500, self.root.winfo_screenheight() - 130)
        self.root.geometry("%dx%d+5+5" % (ww, wh))
        self.root.title("Flipper")
        self.winsize = (self.root.winfo_width(), self.root.winfo_height())

        f1 = tk.Frame(self.root)
        self.lname1 = tk.Label(f1)
        self.lname2 = tk.Label(f1)
        self.lname1.pack(side = "left", anchor = "c", fill = "x", expand = 1)
        self.lname2.pack(side = "left", anchor = "c", fill = "x", expand = 1)
        f1.pack(side = "top", fill = "x", expand = 0)

        self.linfo = tk.Label(self.root, text = "info", bg = "white", anchor = "w")
        self.linfo.pack(side = "top", fill = "x", expand = 1)
        
        self.pimg1 = ImageTk.PhotoImage("RGB")
        self.limg = tk.Label(self.root, image = self.pimg1, cursor = "crosshair", bg="red", borderwidth=0)
        self.limg.pack(side = "top", fill = "none", expand = 0)
        
        self.lcursor = tk.Label(self.root, text="abc", justify="left", bg="white", anchor="w")
        self.lcursor.pack(side = "top", anchor = "w", fill = "x", expand = 1)
        
        linstr = tk.Label(self.root, text=instructions, justify="left", bg="white")
        linstr.pack(side = "top", anchor = "w", fill = "x", expand = 0)

        # initialize everything

        self.imname1 = imn1
        self.imname2 = imn2
        self.name1 = os.path.basename(self.imname1)
        self.name2 = os.path.basename(self.imname2)
        self.fullim1 = Image.open(self.imname1)
        self.fullim2 = Image.open(self.imname2)
        self.zoom = 1 # orig size
        self.cropsize = (100, 100)
        self.croppos = (0, 0)
        self.offset = (0, 0)
        self.diff = False
        self.cursor = (0, 0)
        self.blinkoff()
        self.oldbbox1 = (0, 0, 0, 0)
        self.oldbbox2 = (0, 0, 0, 0)
        self.update()
        
        # key bindings
        self.root.bind("<Configure>", lambda e: self.winresize(e))
        #self.root.bind_all("<ButtonRelease-1>", lambda e: self.checkwinresize())
        self.root.bind("q", lambda e: self.quit())
        self.root.bind("Q", lambda e: self.quit())
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<Left>",  lambda e: self.changeoffset(( 1,  0)))
        self.root.bind("<Right>", lambda e: self.changeoffset((-1,  0)))
        self.root.bind("<Up>",    lambda e: self.changeoffset(( 0,  1)))
        self.root.bind("<Down>",  lambda e: self.changeoffset(( 0, -1)))
        self.root.bind("z",  lambda e: self.changezoom(1))
        self.root.bind("x",  lambda e: self.changezoom(-1))
        self.root.bind("<MouseWheel>", lambda e: self.mousewheel(e))
        self.root.bind("c",  lambda e: self.center())
        self.root.bind("<KeyPress-Tab>", lambda e: self.blinkon())
        self.root.bind("<KeyRelease-Tab>", lambda e: self.blinkoff())
        self.root.bind("<KeyPress-Shift_L>", lambda e: self.diffon())
        self.root.bind("<KeyRelease-Shift_L>", lambda e: self.diffoff())
        #self.root.bind("a", lambda e: self.diffon())
        #self.root.bind("s", lambda e: self.diffoff())
        #self.root.bind("<KeyRelease>",  lambda e: self.update())
        #self.root.bind("<KeyPress>",  lambda e: sys.stdout.write("key:'%s'\n" % e.keysym))
        #self.root.bind("<KeyRelease>",  lambda e: sys.stdout.write("key up:'%s'\n" % e.keysym))

        # mouse bindings for dragging and wheel
        self.limg.bind("<Button-1>", lambda e: self.buttondown(e))
        self.limg.bind("<B1-Motion>", lambda e: self.buttonmove(e))
        
        # and for motion
        self.limg.bind("<Motion>", lambda e: self.updatecursor(e))

        # go!
        self.root.mainloop()

    # end of __init__


    def quit(self):
        for f in glob.glob('tmp-flip-*.p?m'):
            os.remove(f)
        self.root.destroy()
        
    def winresize(self, e):
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if (w, h) != self.winsize:
            self.winsize = (w, h)
            self.maxcropsize = (w - 8, h - 107)
            self.setcropsize()

    def setcropsize(self):
        cs = t2div(self.maxcropsize, self.zoom)
        self.cropsize = boundtuple(cs, (30, 30), self.fullim1.size)
        self.changecroppos((0, 0))
        self.update()

    def changeoffset(self, doffs):
        if self.blink: # shift second image
            self.offset = t2add(self.offset, doffs)
        else: # shift first image
            self.offset = t2sub(self.offset, doffs)
            self.changecroppos(doffs)
        self.update()

    def changecroppos(self, dpos):
        self.setcroppos(t2add(self.croppos, dpos))

    def changecursor(self, dc):
        self.cursor = t2add(self.cursor, dc)
        self.showpos()
            
    def setcroppos(self, cpos):
        oldcroppos = self.croppos
        self.croppos = boundtuple(cpos, (0, 0),
                                  t2sub(self.fullim1.size, self.cropsize))
        self.changecursor(t2sub(self.croppos, oldcroppos))
        self.update()

    def center(self):
        xy = self.cursor
        #print("centering on %d %d" % xy)
        halfcs = t2div(self.cropsize, 2)
        self.setcroppos(t2sub(xy, halfcs))

    def changezoom(self, dz):
        newzoom = max(1, min(9, self.zoom + dz))
        if newzoom != self.zoom:
            oldhalfcs = t2div(self.cropsize, 2)
            self.zoom = newzoom
            self.setcropsize() # this actually changes the zoom
            newhalfcs = t2div(self.cropsize, 2)
            self.changecroppos(t2sub(oldhalfcs, newhalfcs)) # adjust center

    def mousewheel(self, event):
        self.changezoom(event.delta//120)

    def blinkon(self):
        self.lname1.configure(bg = "white")
        self.lname2.configure(bg = "#ffb")
        self.blink = True
        if self.pimg2 == None:
            self.update()
        self.limg.configure(image = self.pimg2)
        self.showpos()

    def blinkoff(self):
        self.lname1.configure(bg = "#ffb")
        self.lname2.configure(bg = "white")
        self.blink = False
        if self.pimg1 == None:
            self.update()
        self.limg.configure(image = self.pimg1)
        self.showpos()

    def diffon(self):
        if self.diff:
            return          # important if Shift is used because of key repeat!!!!
        self.diff = True
        self.update()

    def diffoff(self):
        if not self.diff:
            return
        self.diff = False
        self.update()
        
    def buttondown(self, event):
        self.oldcroppos = self.croppos
        self.oldoffset = self.offset
        self.down = event.x, event.y

    def buttonmove(self, event):
        dxy = t2div(t2sub(self.down, (event.x, event.y)), self.zoom)
        if self.diff:
            self.offset = t2add(self.oldoffset, dxy)
        else:
            self.setcroppos(t2add(self.oldcroppos, dxy))
        self.update()
        
    def updatecursor(self, e):
        self.cursor = t2add(self.croppos, t2div((e.x, e.y), self.zoom))
        self.showpos()
        
    def showpos(self):
        if self.blink:
            xy = t2add(self.cursor, self.offset)
            pix = getpixel(self.fullim2, xy)
        else:
            xy = self.cursor
            pix = getpixel(self.fullim1, xy)
        self.lcursor.configure(text = "%4d,%4d: %s" % (xy + (pix,)))

    # update images
    def update(self):
        cp1 = self.croppos
        cp2 = t2add(cp1, self.offset)
        z = self.zoom
        #cw = t2div(self.cropsize, z)
        cw =  self.cropsize # NEW: already reflects zoom!
        cz = t2mul(cw, z)
        bbox1 = cp1 + t2add(cp1, cw) # '+' concatenates tuples
        bbox2 = cp2 + t2add(cp2, cw)
        if self.oldbbox1 != bbox1:
            self.oldbbox1 = bbox1
            self.img1 = self.fullim1.crop(bbox1)
            self.pimg1 = None
        if self.oldbbox2 != bbox2:
            self.oldbbox2 = bbox2
            self.img2 = self.fullim2.crop(bbox2)
            self.pimg2 = None

        if self.diff: # residual
            residscale = 3
            resid = ImageChops.subtract(self.img1, self.img2, 1.0/residscale, 128)
            if z > 1:
                resid = resid.resize(cz)
            self.presid = ImageTk.PhotoImage(resid)
            self.limg.configure(image = self.presid)
        else:
            if self.blink:
                im2 = self.img2 if z == 1 else self.img2.resize(cz)
                self.pimg2 = ImageTk.PhotoImage(im2)
                self.limg.configure(image = self.pimg2)
            else:
                im1 = self.img1 if z == 1 else self.img1.resize(cz)
                self.pimg1 = ImageTk.PhotoImage(im1)
                self.limg.configure(image = self.pimg1)
        #update labels:
        t1 = self.name1 + "  %+d%+d " % cp1
        t2 = self.name2 + "  %+d%+d " % cp2
        t3 = "offset:  dx = %+d  dy = %+d" % self.offset
        t3 += "    zoom %d:1" % self.zoom
        t3 += "    window: %dx%d" % cw
        self.lname1.configure(text = t1)
        self.lname2.configure(text = t2)
        self.linfo.configure(text = t3)
        self.showpos()
        

# utility functions

# t2 - operations on tuples of 2 ints
def t2add(a, b):
    a1, a2 = a
    b1, b2 = b
    return (a1+b1, a2+b2)

def t2sub(a, b):
    a1, a2 = a
    b1, b2 = b
    return (a1-b1, a2-b2)

def t2mul(a, s):
    a1, a2 = a
    return (int(s*a1), int(s*a2))

def t2div(a, s):
    a1, a2 = a
    return (int(a1/s), int(a2/s))

def boundtuple(a, lo, hi):
    a1, a2 = a
    lo1, lo2 = lo
    hi1, hi2 = hi
    return (max(lo1, min(hi1, a1)),
            max(lo2, min(hi2, a2)))

def getpixel(im, xy):
    w, h = im.size
    x, y = xy
    if 0 <= x < w and 0 <= y < h:
        return im.getpixel(xy)
    else:
        return ""


# run the whole thing
flipper(sys.argv[1], sys.argv[2])
