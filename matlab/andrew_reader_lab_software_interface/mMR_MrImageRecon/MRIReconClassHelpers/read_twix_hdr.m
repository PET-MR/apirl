function [prot,rstraj] = read_twix_hdr(fid)
% function to read raw data header information from siemens MRI scanners 
% (currently VB and VD software versions are supported and tested).
%
% Author: Philipp Ehses (philipp.ehses@tuebingen.mpg.de), Mar/11/2014
      
    nbuffers = fread(fid, 1,'uint32');
    
    prot = [];
    for b=1:nbuffers
        namesz  = 0;
        byte = 1;
        while byte~=0 % look for NULL-character
            byte   = fread(fid, 1, 'uint8');
            namesz = namesz+1;
        end
        fseek(fid,-namesz,'cof');
        bufname        = fread(fid,namesz,'char=>char').';
        bufname = deblank(bufname); %added mehranian 15102016
        buflen         = fread(fid, 1,'uint32');
        buffer         = fread(fid, buflen, 'char=>char').';
        buffer         = regexprep(buffer,'\n\s*\n',''); % delete empty lines
        a= parse_buffer(buffer,bufname);
        prot.(bufname) =a;
    end
    
    if nargout>1
        rstraj = [];
        if prot.Meas.alRegridMode(1)>1
            ncol           = prot.Meas.alRegridDestSamples(1);
            dwelltime      = prot.Meas.aflRegridADCDuration(1)/ncol;
            gr_adc         = zeros(1,ncol,'single');
            time_adc       = prot.Meas.alRegridDelaySamplesTime(1) + dwelltime * (0.5:ncol);
            ixUp           = time_adc <= prot.Meas.alRegridRampupTime(1);
            ixFlat         = (time_adc <= prot.Meas.alRegridRampupTime(1)+prot.Meas.alRegridFlattopTime(1)) & ~ixUp;
            ixDn           = ~ixUp & ~ixFlat;
            gr_adc(ixFlat) = 1;
            if prot.Meas.alRegridMode(1) == 2  
                % trapezoidal gradient
                gr_adc(ixUp)   = time_adc(ixUp)/prot.Meas.alRegridRampupTime(1);
                gr_adc(ixDn)   = 1 - (time_adc(ixDn)-prot.Meas.alRegridRampupTime(1)-prot.Meas.alRegridFlattopTime(1))/prot.Meas.alRegridRampdownTime(1);
            elseif prot.hdr.Meas.alRegridMode(1) == 4  
                % sinusoidal gradient
                gr_adc(ixUp)   = sin(pi/2*time_adc(ixUp)/prot.Meas.alRegridRampupTime(1));
                gr_adc(ixDn)   = sin(pi/2*(1+(time_adc(ixDn)-prot.Meas.alRegridRampupTime(1)-prot.Meas.alRegridFlattopTime(1))/prot.Meas.alRegridRampdownTime(1)));
            else
                warning('regridding mode unknown');
                return;
            end
            rstraj = (cumsum(gr_adc(:)) - ncol/2)/sum(gr_adc(:));
            rstraj = rstraj - rstraj(ncol/2+1);
        end
    end
    
end        
    
function prot = parse_buffer(buffer,bufname)
    [ascconv,xprot] = regexp(buffer,'### ASCCONV BEGIN[^\n]*\n(.*)\s### ASCCONV END ###','tokens','split');
    xprot = xprot{:};

    if ~isempty(ascconv)
        prot = parse_ascconv(ascconv{:}{:});
    else
        prot = struct();
    end
         
    if ~isempty(xprot)
        xprot = parse_xprot(xprot);
        if isstruct(xprot)
            name   = cat(1,fieldnames(prot),fieldnames(xprot));
            val    = cat(1,struct2cell(prot),struct2cell(xprot));
            [~,ix] = unique(name);
            prot   = cell2struct(val(ix),name(ix));
        end
    end
end

function xprot = parse_xprot(buffer)
    xprot = [];
    tokens = regexp(buffer, '<Param(?:Bool|Long|String)\."(\w+)">\s*{([^}]*)','tokens');
    tokens = [tokens, regexp(buffer, '<ParamDouble\."(\w+)">\s*{\s*(<Precision>\s*[0-9]*)?\s*([^}]*)','tokens')];
    for m=1:numel(tokens)
        name         = char(tokens{m}(1));
        % field name has to start with letter
        if (~isletter(name(1)))
            name = strcat('x', name);
        end

        value = char(strtrim(regexprep(tokens{m}(end), '("*)|( *<\w*> *[^\n]*)', '')));
        value = regexprep(value, '\s*', ' ');

        if isempty(value)
            value = [];
        else
            tmp = str2num(value);
            if ~isempty(tmp)
                value = tmp;
            end
        end   
        
        xprot.(name) = value;
    end
end

function mrprot = parse_ascconv(buffer)  
    mrprot = [];    
    % [mv] was: vararray = regexp(buffer,'(?<name>\S*)\s*=\s(?<value>\S*)','names');
    vararray = regexp(buffer,'(?<name>\S*)\s*=\s*(?<value>\S*)','names');
    
    for var=vararray
        value = str2num(var.value);
        if isempty(value)
            value = var.value;
        end
        
        % now split array name and index (if present)
        v = regexp(var.name,'(?<name>\w*)\[(?<ix>[0-9]*)\]|(?<name>\w*)','names');

        cnt = 0;
        tmp = cell(2,numel(v));

        breaked = false;
        for k=1:numel(v)
            cnt = cnt+1;
            tmp{1,cnt} = '.';
            if ~isletter(v(k).name(1))
                breaked = true;
                break;
            end
            tmp{2,cnt} = v(k).name;
            if ~isempty(v(k).ix)
                cnt = cnt+1;
                tmp{1,cnt} = '{}';
                tmp{2,cnt}{1} = 1 + str2double(v(k).ix);
            end
        end
        if ~breaked
            S = substruct(tmp{:});
            mrprot = subsasgn(mrprot,S,value);
        end
    end 
end

