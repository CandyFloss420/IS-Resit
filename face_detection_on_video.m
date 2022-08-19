clc, clear, close all;

%debug mode: Set to 1 if debug mode is active
debug = 0;

%counters
num_frames = 0;    %number of frames in video
num_frames_hit = 0;%number of frames with face detected
facefoundin = [];  %frame number array for hits

%load video in workspace
video_input = VideoReader('kelseylighting.avi');  %change kelseystill.avi to 
                                               %kelseymotion.avi or 
                                               %kelseylighting.avi

%extract frames from loaded video
frames = readFrame(video_input);

% Create the cascade detector object.
face_detector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
bbox   = step(face_detector, frames);

% Draw the returned bounding box around the detected face.
video_out = insertShape(frames, 'Rectangle', bbox);

%converting detected box to points
bbox_points = bbox2points(bbox(1,:));

%get feature points of the face - this detector works in grayscale only
face_points = detectMinEigenFeatures(rgb2gray(frames), 'ROI', bbox);

%point tracker tracks the faces location
point_tracker = vision.PointTracker('MaxBidirectionalError', 2);

%get the location of face points in X,Y coordinates
face_points = face_points.Location;

%initialise point tracker at the face point location
initialize(point_tracker, face_points, video_out);

%create video player to play video
left = 100;
bottom = 100;
width = size(video_out, 2);
height = size(video_out, 1);
video_player = vision.VideoPlayer('Position', [left bottom width height]);

%compare the change of the location of face points
previous_points = face_points;

while hasFrame(video_input)
    num_frames = num_frames+1;
    
    frames = readFrame(video_input);
    
    %search for face in current frame
    [face_points, detected] = step(point_tracker, frames);

    %Counters annd debug code control
    if detected == 1
        num_frames_hit = num_frames_hit+1;
    else
        disp("Frame not detected, number " + num_frames);
        if debug == 1
            %for each frame, pause and let the user see the frame. 
            %wait for users response before continuing to the next frame.
            imshow(frames);
            uiwait;
            uiresume;
        end
    end
    
    %prepare face point matirx to detectand calculate the rotation of the face
    new_points = face_points(detected, :);
    old_points = previous_points(detected, :);

    %display number of points detected in frame (statictics)
    disp("Face points detected in frame "+num_frames +" = " + size(new_points,1));
    
    %check if face orientation has changed. If it has changed, rotate the polygon accordingly
    if size(new_points, 1) >= 2
        %geometry transform the rectangle
        [transform_rectangle, old_points, new_points] = estimateGeometricTransform(old_points, new_points, 'similarity', 'MaxDistance', 4);
        bbox_points = transformPointsForward(transform_rectangle, bbox_points);
        %reshape_rectangle = reshape(bbox_points, 1, []); %REMOVE: BUG CODE
        %insert face box rectangle in video frame
        video_out = insertShape(frames, 'Polygon', bbox_points, 'LineWidth', 5);
        %video_out = insertShape(frames, 'Polygon', reshape_rectangle, 'LineWidth', 5);
        video_out = insertMarker(video_out, new_points, '+', 'Color', 'white'); %REMOVE: BUG CODE
        
        %reset pointers ready for next iteration
        previous_points = new_points;
        setPoints(point_tracker, previous_points);
    end
    
    %display video with detected frames
    step(video_player, video_out)
end
%display final statictics
disp("Number of frames in input video = " + num_frames);
disp("Number of frames with face detected = " + num_frames_hit);

%release video player
release(video_player);