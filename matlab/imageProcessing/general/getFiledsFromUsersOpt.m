
function Obj = getFiledsFromUsersOpt(Obj,opt)
vfields = fieldnames(opt);
if isstruct(Obj)
    prop = fieldnames(Obj);
else
    prop = properties(Obj);
end
for i = 1:length(vfields)
    field = vfields{i};
    if sum(strcmpi(prop, field )) > 0
        Obj.(field) = opt.(field);
    end
end
end