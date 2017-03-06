
function SaveMichelogram (MyMichelogram, NombreArch)

FID = fopen(NombreArch, 'w');
% El Michelogram que usamos en MATLAB es de cuatro dimensiones. La tercera
% y cuarta dimensión representan los rings1 y 2 respectivamente. Cada combinación de rings tiene
% un sinograma, los vamos guardando en un archivo, yendo sinograma por
% sinograma siguiendo el barrido columna - fila. O sea recorro primero el
% ring1 y después avanzo al ring2. Cada Sinograma esta representado por una
% matriz que contiene en las filas el ángulo Phi y en las columnas las
% variables R.
for i = 1 : size(MyMichelogram,4)       % Recorro el ring 2
    for j = 1 : size(MyMichelogram,3)   % Recorro el ring 1
        Sino2D = MyMichelogram(:,:,j,i)';   % Hago la traspuesta porque cuando lo llamo
                                            % como vector recorre fila -
                                            % columna y yo quiero que sea a
                                            % la inversa.
        fwrite(FID, Sino2D(:), 'float')
    end
end

fclose(FID);