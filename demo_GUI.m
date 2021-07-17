%% gui demo
% Author: Mahmoud Afifi
% Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
% Please cite our paper:
% Mahmoud Afifi,  Konstantinos G. Derpanis, Björn Ommer, and Michael S
% Brown. Learning Multi-Scale Photo Exposure Correction, In CVPR 2021.
%%

function varargout = demo_GUI(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @demo_GUI_OpeningFcn, ...
    'gui_OutputFcn',  @demo_GUI_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function demo_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
global net
handles.browse_btn.Enable = 'Off';
handles.status.String = 'Loading...';pause(0.001);
net = load(fullfile('models','model.mat'));
handles.browse_btn.Enable = 'On';
handles.status.String = 'Ready!';pause(0.001);
handles.status.String = '';pause(0.001);
addpath('bgu');
addpath('exFusion');
handles.output = hObject;
guidata(hObject, handles);
handles.save_btn.Enable = 'Off';


function varargout = demo_GUI_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

function save_btn_Callback(hObject, eventdata, handles)
global Path_Name File_Name output_final I Original_I
[~,name,ext] = fileparts(File_Name);
outFile_Name = [name '_enhanced' ext];
[file,path,~] = uiputfile({'*.jpg';'*.png';'*.jpeg';'*.*'},...
    'Save Image',fullfile(Path_Name,outFile_Name));
if file ~=0
    handles.status.String = 'Processing...';pause(0.001);
    output_s = double(imresize(output_final,[200,200]));
    I_s = double(imresize(I,[200,200]));
    results = computeBGU(I_s, rgb2luminance(I_s), output_s, [], ...
        Original_I, rgb2luminance(Original_I));
    output = results.result_fs;
    imwrite(output, fullfile(path,file));
    handles.status.String = 'Done!';
    pause(0.01); handles.status.String = '';
end

function browse_btn_Callback(hObject, eventdata, handles)
global Path_Name File_Name Original_I I net output output_pp output_p ...
    output_final pad_factor


[File_Name, Path_Name] = uigetfile({'*.jpg';'*.png';'*.jpeg'},...
    'Select input image','example_images');

if File_Name ~=0
    Original_I = im2double(imread(fullfile(Path_Name,File_Name)));
    handles.status.String = 'Loading image...';pause(0.001);
    %% check image size
    sz = size(Original_I);
    inSz = 512;
    S = [handles.s_1.Value, handles.s_2.Value, ...
        handles.s_3.Value, handles.s_4.Value];
    I = imresize(Original_I,inSz/max(sz));
    pad_factor = [inSz-size(I,1) inSz-size(I,2)];
    I = padarray(I, pad_factor,'replicate','pre');
    Image = pre_process_img(I,4,S);
    axes(handles.image);
    I = I(pad_factor(1)+1:end,pad_factor(2)+1:end,:);
    imshow(I);
    pause(0.2);
    [output_pp, output] = correct_image(handles, Image, I, net.net, pad_factor);
    output_p = output_pp;
    output_final = output_pp .* handles.intensity.Value + ...
        I .* (1 - handles.intensity.Value);
    imshow(output_final);
    handles.save_btn.Enable = 'On';
    handles.status.String = 'Done!';pause(0.001);
    handles.status.String = '';pause(0.001);
end

function [output_pp, output] = correct_image(handles, Image, I, net, pad_factor)
if canUseGPU == 1
    handles.status.String = 'Processing on GPU...';
    output = gather(extractdata(predict(net,gpuArray(dlarray(Image,...
        'SSCB')))))/255;
else
    handles.status.String = 'Processing on CPU...';
    output = extractdata(predict(net,dlarray(Image,'SSCB')))/255;
end
output = output(pad_factor(1)+1:end,pad_factor(2)+1:end,1:3);

if handles.fusion_on.Value == 1
    handles.status.String = 'Applying fusion...';
    Out = zeros(size(output,1),size(output,2),size(output,3),2);
    Out(:,:,:,1) = I;
    Out(:,:,:,2) = output;
    output_pp = exposure_fusion(Out,[1 1 1]);
else
    output_pp = output;
end

if handles.contrast_on.Value == 1
    handles.status.String = 'Adjusting contrast...';
    output_pp = histAdjust(output);
end

function intensity_Callback(hObject, eventdata, handles)
global I output_pp output_p output_final
if handles.contrast_off.Value == 1
    output_final = output_p .* handles.intensity.Value + ...
        I .* (1 - handles.intensity.Value);
else
    output_final = output_pp .* handles.intensity.Value + ...
        I .* (1 - handles.intensity.Value);
end

axes(handles.image);
imshow(output_final)

function s_1_Callback(hObject, eventdata, handles)
global I output output_final pad_factor output_pp output_p net
S = [handles.s_1.Value, handles.s_2.Value, ...
    handles.s_3.Value, handles.s_4.Value];
I = padarray(I, pad_factor,'replicate','pre');
Image = pre_process_img(I,4,S);
I = I(pad_factor(1)+1:end,pad_factor(2)+1:end,:);
[output_pp, output] = correct_image(handles, Image, I, net.net, pad_factor);
output_p = output_pp;
output_final = output_pp .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);
handles.status.String = '';pause(0.001);

function s_3_Callback(hObject, eventdata, handles)
global I output output_final pad_factor output_pp output_p net
S = [handles.s_1.Value, handles.s_2.Value, ...
    handles.s_3.Value, handles.s_4.Value];
I = padarray(I, pad_factor,'replicate','pre');
Image = pre_process_img(I,4,S);
I = I(pad_factor(1)+1:end,pad_factor(2)+1:end,:);
[output_pp, output] = correct_image(handles, Image, I, net.net, pad_factor);
output_p = output_pp;
output_final = output_pp .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);
handles.status.String = '';pause(0.001);

function s_2_Callback(hObject, eventdata, handles)
global I output output_final pad_factor output_pp output_p net
S = [handles.s_1.Value, handles.s_2.Value, ...
    handles.s_3.Value, handles.s_4.Value];
I = padarray(I, pad_factor,'replicate','pre');
Image = pre_process_img(I,4,S);
I = I(pad_factor(1)+1:end,pad_factor(2)+1:end,:);
[output_pp, output] = correct_image(handles, Image, I, net.net, pad_factor);
output_p = output_pp;
output_final = output_pp .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);
handles.status.String = '';pause(0.001);


function s_4_Callback(hObject, eventdata, handles)
global I output output_final pad_factor output_pp output_p net
S = [handles.s_1.Value, handles.s_2.Value, ...
    handles.s_3.Value, handles.s_4.Value];
I = padarray(I, pad_factor,'replicate','pre');
Image = pre_process_img(I,4,S);
I = I(pad_factor(1)+1:end,pad_factor(2)+1:end,:);
[output_pp, output] = correct_image(handles, Image, I, net.net, pad_factor);
output_p = output_pp;
output_final = output_pp .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);
handles.status.String = '';pause(0.001);

function fusion_on_Callback(hObject, eventdata, handles)
global output output_p output_final I
handles.status.String = 'Applying fusion...';
Out = zeros(size(output,1),size(output,2),size(output,3),2);
Out(:,:,:,1) = I;
Out(:,:,:,2) = output;
output_p = exposure_fusion(Out,[1 1 1]);
output_final = output_p .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);
handles.status.String = 'Done!';pause(0.001);
handles.status.String = '';pause(0.001);

function fusion_off_Callback(hObject, eventdata, handles)
global output output_p output_final I
output_p = output;
output_final = output_p .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);

function contrast_off_Callback(hObject, eventdata, handles)
global output_pp output_p output_final I
output_pp = output_p;
output_final = output_pp .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);

function contrast_on_Callback(hObject, eventdata, handles)
global output_pp output_p output_final I
handles.status.String = 'Adjusting contrast...';
output_pp = histAdjust(output_p);
output_final = output_pp .* handles.intensity.Value + ...
    I .* (1 - handles.intensity.Value);
axes(handles.image);
imshow(output_final);
handles.status.String = 'Done!';pause(0.001);
handles.status.String = '';pause(0.001);


%% create fncs
function s_4_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function s_3_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function s_2_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function s_1_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function intensity_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
