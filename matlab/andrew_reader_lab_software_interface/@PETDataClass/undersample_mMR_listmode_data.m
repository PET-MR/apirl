
function [totalPrompts,totalRandoms, totalWords, outFileHdr, output_listmode_file] = undersample_mMR_listmode_data(ObjData,input_listmode_file,countReductionFractor,chunk_size_events, numRealizations)

if nargin<4
   chunk_size_events = 100e6;
   numRealizations = 1;
end
    
if nargin < 5
    numRealizations = 1;
end

% Read the list mode file.
fid = fopen(input_listmode_file, 'r');
if fid == -1
    error('Error: List-mode binary file not found.');
end

% Create the ouput files:
for n = 1 : numRealizations
    output_listmode_file{n} = [input_listmode_file(1:end-2) '-' num2str(100-countReductionFractor*100, '%.1f') '-r' num2str(n) '.l'];
    fid2{n} = fopen(output_listmode_file{n},'a');
    
    totalPrompts(n) = 0;
    totalRandoms(n) = 0;
    totalWords(n) = 0;
end
    
countsFraction = 1-countReductionFractor;

while(~feof(fid))
    data =fread(fid,chunk_size_events,'*uint32');
    P = ~bitor(bitshift(data,-31),0) & bitand(bitshift(data,-30),2^1-1);   % bit 31=0, bit 30=1 prompts, bit30=1 delays

    idx_prompts = P==1;
    idx_randoms = P==0;

    nPromptsPerChunk = sum(idx_prompts);
    nRandomsPerChunk = sum(idx_randoms);
    
    nPromptsPerChunk_this_real = floor(countsFraction*nPromptsPerChunk);
    nRandomsPerChunk_this_real = floor(countsFraction*nRandomsPerChunk);
    
    % Generate the realization
    prompts = find(idx_prompts);
    randoms = find(idx_randoms);
    for n = 1 : numRealizations    
        
        b = randperm(nPromptsPerChunk,nPromptsPerChunk_this_real);
        idx_prompts_this_real = prompts(b);

        b = randperm(nRandomsPerChunk,nRandomsPerChunk_this_real);
        idx_randoms_this_real = randoms(b);

        idx_events_this_real = sort([idx_prompts_this_real; idx_randoms_this_real]);

        data_this_real = data(idx_events_this_real);
        
        fwrite(fid2{n},data_this_real,'uint32');
        
        totalPrompts(n) = totalPrompts(n) + length(idx_prompts_this_real);
        totalRandoms(n) = totalRandoms(n) + length(idx_randoms_this_real);
        totalWords(n) = totalWords(n) + length(data_this_real);
    
        % Remove events from prompt data:
        data(idx_events_this_real) = [];
        idx_prompts(idx_events_this_real) = [];
        idx_randoms(idx_events_this_real) = [];
        nPromptsPerChunk = sum(idx_prompts);
        nRandomsPerChunk = sum(idx_randoms);
        prompts = find(idx_prompts);
        randoms = find(idx_randoms);
    end

end
fclose(fid);

for n = 1 : numRealizations 
    % Creat interfile header
    inFileHdr = [input_listmode_file(1:end-1) 'hdr'];
    fid3 = fopen(inFileHdr,'r');

    if fid3==-1
        error('Error: List-mode header file was not found.');
    end
    outFileHdr{n} = [output_listmode_file{n}(1:end-2) '.hdr'];
    fid4 = fopen(outFileHdr{n},'w');

    while (true)
        line_txt = fgetl(fid3);
        if (line_txt == -1)
            break;
        end
        line_txt = regexprep(line_txt, ';.*$', '');
        token = '%total listmode word counts :=';
        totalWordsIndx = strfind(line_txt,token);
        if ~isempty(totalWordsIndx)
            line_txt(totalWordsIndx+length(token):end) = [];
            line_txt = [line_txt num2str(totalWords(n))];
        end
        fprintf(fid4,'%s\n',line_txt);
    end
    fclose(fid2{n});
    fclose(fid3);
    fclose(fid4);
end




