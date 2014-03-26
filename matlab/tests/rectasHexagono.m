% Grafico curvas de cabezal
distToCabezal = 360;
y0 = distToCabezal / sind(60);
x1 = -360:0;
y1 = tand(30)*x1+y0;
x2 = 0:360;
y2 = tand(-30)*x2+y0;
x3 = 0:360;
y3 = tand(30)*x3-y0;
x4 = -360:0;
y4 = tand(-30)*x4-y0;
x5 = -360;
y5 = -203:203;
x6 = 360;
y6 = -203:203;
plot(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6);