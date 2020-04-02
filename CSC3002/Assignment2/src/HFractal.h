#ifndef HFRACTAL_H
#define HFRACTAL_H

#include "lib/StanfordCPPLib/graphics/gwindow.h"
#include <cmath>

/* Function prototypes */

/*
 * Function: drawHFractal
 * Usage: drawHFractal(gw, x, y, size, order);
 * -------------------------------------------
 * Draws a fractal diagram consisting of an H in which each additional
 * fractal layer draws half-size fractals at the four endpoints of each H.
 */

void drawHFractal(GWindow & gw, double x, double y, double size, int order);

/* Test Function of Q3 */

void hFractal();

#endif // HFRACTAL_H
