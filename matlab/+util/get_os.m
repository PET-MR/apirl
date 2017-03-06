function [sys, pathBar] = get_os()

pathBar = filesep;
if(strcmp(computer(), 'GLNXA64'))
    sys = 'linux';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    sys = 'windows';
else
    disp('OS not compatible');
    sys = '';
end

end  % function os()