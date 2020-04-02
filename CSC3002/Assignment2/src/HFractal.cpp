/*
 * File: HFractal.cpp
 * ------------------
 * This program draws an H-fractal on the graphics window.int main() {
 */

#include "HFractal.h"

/*
 * Function: drawHFractal
 * Usage: drawHFractal(gw, x, y, size, order);
 * -------------------------------------------
 * Draws a fractal diagram consisting of an H in which each additional
 * fractal layer draws half-size fractals at the four endpoints of each H.
 */

void drawHFractal(GWindow & gw, double x, double y, double size, int order) {

   if (order == 0){
       double hsize = size / 2;
       GPoint p1(x - hsize, y - hsize),
              p2(x - hsize, y + hsize),
              p3(x - hsize, y),
              p4(x + hsize, y),
              p5(x + hsize, y - hsize),
              p6(x + hsize, y + hsize);

       gw.drawLine(p1, p2);
       gw.drawLine(p3, p4);
       gw.drawLine(p5, p6);
   }

   else {
       double parent = size / ( 2 - std::pow(2, -order) ); // size of parent H
       double halfParent = parent / 2;
       double child = size - parent; // size of the children fractals

       /*if (order & 1) gw.setColor("Blue"); // odd order
       else gw.setColor("Red");*/

       drawHFractal(gw, x, y, parent, 0); // parent H in the middle
       drawHFractal(gw, x - halfParent, y - halfParent, child, order - 1); // top left child
       drawHFractal(gw, x + halfParent, y - halfParent, child, order - 1); // top right child
       drawHFractal(gw, x - halfParent, y + halfParent, child, order - 1); // bottom left child
       drawHFractal(gw, x + halfParent, y + halfParent, child, order - 1); // bottom right child
   }
}

/* Test Function of Q3 */

void hFractal() {
   GWindow gw(1000, 1000);
   gw.setTitle("H Fractal");
   gw.setColor("Black");
   gw.setExitOnClose(true);

   double xc = gw.getWidth() / 2;
   double yc = gw.getHeight() / 2;
   drawHFractal(gw, xc, yc, 900, 4);

   gw.setVisible(true);
   gw.requestFocus();
}
