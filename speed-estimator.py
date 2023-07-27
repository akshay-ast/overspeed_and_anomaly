#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
from datetime import datetime
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst, GObject
from ctypes import *
import time
import sys
import math
import threading
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import cv2
import pyds
import numpy as np
import json
import pymongo
from pymongo import MongoClient
from botocore.exceptions import NoCredentialsError
from db_utils import SECRET_KEY, BUCKET_NAME, ACCESS_KEY
from db_utils import upload_file_to_s3, get_database, push_to_db
perf_data = None
from datetime import datetime, timedelta

started=False
stop_time =  None
saveVideo = None
image = None

# Initialize constants
MAX_DISPLAY_LEN=64
PGIE_CLASS_ID_SACK = 0
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
# classes_dict = {0:"Car", 1: "Motorcycle", 2:"Bus", 3:"Truck"}
classes_dict={2:'Car',7:'Truck'}
LINE_SPACING = 10.0  # 10 meters distance between lines
MPS_TO_KMH = 3.6

# Initialize global variables
object_speeds = {}
lock = threading.Lock()
speed_kmh = 0
db_collection = get_database()
vehicle_coords = []
# nvanlytics_src_pad_buffer_probe  will extract metadata received on nvtiler sink pad
# and update params for drawing rectangle, object information etc.

def push_to_db(frame, db_collection, img_name, alert):    
    img_encode = cv2.imencode('.jpg', frame)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tobytes()
    url = upload_file_to_s3(str_encode,img_name +'.jpg')
    print("url:", url)
    alert["Image"] = url
    db_collection.insert_one(alert)
    print(f"Data pushed. Image uploaded to {url}")
    return True

def nvanalytics_src_pad_buffer_probe(pad,info,u_data):
    global object_speeds, lock, speed_kmh, vehicle_coords
    global MPS_TO_KMH
    global SECRET_KEY, BUCKET_NAME, ACCESS_KEY, db_collection,started,stop_time,saveVideo,frame_copy
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

            if obj_meta and obj_meta.obj_user_meta_list:
                l_user = obj_meta.obj_user_meta_list
                while l_user:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                        analytics_obj_info = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)


                        # Check if the object has crossed any lines
                        if len(analytics_obj_info.lcStatus) > 0:
                            object_id = obj_meta.object_id
                            line_crossed = analytics_obj_info.lcStatus
                            current_time = time.time()                            

                            lock.acquire()

                            # If object is not in the dictionary add it
                            if object_id not in object_speeds:
                                object_speeds[object_id] = {
                                    'last_line_crossed': line_crossed,
                                    'last_cross_time': current_time
                                }
                                print("Object Speeds", object_speeds)
                            else:
                                # Calculate the time difference and distance crossed
                                time_difference = current_time - object_speeds[object_id]['last_cross_time']
                                print("Time difference:", time_difference)
                                distance = 10
                                print("Distance:", distance)

                                # Calculate speed in mps and convert to kmh
                                speed_mps = distance / time_difference
                                speed_kmh = speed_mps * MPS_TO_KMH

                                print(f"Object {object_id}: Speed = {speed_kmh:.2f} km/h")

                                # Update object's crossing info
                                object_speeds[object_id]['last_line_crossed'] = line_crossed
                                object_speeds[object_id]['last_cross_time'] = current_time
                                vehicle_coords = obj_meta.rect_params
                            
                            lock.release()
                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        #n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
        # convert python array into numpy array format in the copy mode.
        frame_copy = np.array(n_frame, copy=True, order='C')
        # convert the array into cv2 default color format
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
        frame_number=frame_meta.frame_num
        if frame_number in [0,1,2,3]:
            stop_time = datetime.now()
            print(stop_time)

        l_user=frame_meta.frame_user_meta_list
        num_rects = frame_meta.num_obj_meta                      
        
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        py_nvosd_text_params.display_text = f"Average speed: {speed_kmh:.2f} km/h"

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        if speed_kmh > 10:             
            cv2.putText(frame_copy, text=f"Speed: {speed_kmh:.2f}km/h", org=(50,600), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
            left = int(vehicle_coords.left)
            top = int(vehicle_coords.top)
            right = int(vehicle_coords.left) + int(vehicle_coords.width)
            bottom = int(vehicle_coords.top) + int(vehicle_coords.height)
            # print(left,right,top,bottom)
            now = datetime.now()
            time_1 = now.strftime("%H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            img_name = "vehicle" + str(date) + str(time_1)
            cv2.rectangle(frame_copy,(left,top),(right,bottom),(0,0,255),2)
            cv2.imwrite(f"frames/{img_name}.jpg", frame_copy)
            print(f"Saved image as {img_name}.jpg")
            alert = {
            "Camera": "CCU Road 1",
            "Date": date,
            "Time": time_1,
            "Alert_Type": "Overspeed",
            "Status": "Warning",
                        }
            print(alert)
            push_to_db(frame_copy, db_collection, img_name, alert)
            print("Pushed to database")            
            speed_kmh = 0
        now1 = datetime.now()
        if not started and now1 > stop_time:
            stop_time = datetime.now() + timedelta(hours=3)
            t_stamp = datetime.now()
            t_stamp = t_stamp.strftime('%d-%m-%Y_%H:%M:%S:%f').replace(" ", "_")
            #saveVideo = cv2.VideoWriter('{}Frisk{}.mp4'.format(f"stream{frame_meta.source_id}", t_stamp),
            #        cv2.VideoWriter_fourcc(*'XVID'), 5, (1920, 1080))
            saveVideo = cv2.VideoWriter(t_stamp+ ".mp4",cv2.VideoWriter_fourcc(*'MJPG'),10, (1920, 1080))
            started=True
        if now1 <= stop_time and started:
            video_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGRA2BGR)
            saveVideo.write(video_frame)
            print("writing started")
        if now1 > stop_time and started:
            saveVideo.release()
            started =  False
            print("stopped")

                    
        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        # print("#"*50)

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)


def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main():
    # Check input arguments
    SOURCES = {
    # Stream 0 CCU Road 1
    0:["rtsp://admin:Pass%40123@192.168.100.104:554/Streaming/Channels/101/?transportmode=unicast"],
    # 0:["file:/home/akshay/assert/birla/vehicle/ccu-road-1/ccur1-trim-1.mp4"],
    # Stream 1 New PAC back 1
    # 1:["file:/home/akshay/assert/birla/vehicle/new-pac-back-2/trim-1.mp4"],
    1:["rtsp://admin:Pass%40123@192.168.100.114:554/Streaming/Channels/101/?transportmode=unicast"],
    # Stream 2 New PAC back 2  
    # 2:["file:/home/akshay/assert/birla/vehicle/pa-drum-road-1/padr1-trim-1.mp4"],
    # 2:["rtsp://admin:Pass%40123@192.168.100.115:554/Streaming/Channels/101/?transportmode=unicast"],
    # Stream 2 Material Gate C1
    # 3:["file:/home/akshay/assert/birla/vehicle/ccu-road-1/ccur1-trim-1.mp4"],
    2:["rtsp://admin:Pass%40123@192.168.100.51:554/Streaming/Channels/101/?transportmode=unicast"],
    # Stream 3 Material Gate C3
    # 3:["file:/home/akshay/assert/birla/vehicle/ccu-road-1/ccur1-trim-1.mp4"],
    3:["rtsp://admin:Pass%40123@192.168.100.53:554/Streaming/Channels/101/?transportmode=unicast"],
    # Stream 4 Tech Build Corner 1
    # 5:["file:/home/akshay/assert/birla/vehicle/ccu-road-1/ccur1-trim-1.mp4"],
    4:["rtsp://admin:Pass%40123@192.168.100.82:554/Streaming/Channels/101/?transportmode=unicast"],
    }
    number_sources = len(SOURCES.items())
    global perf_data
    perf_data = PERF_DATA(number_sources)
    #number_sources=len(args)-1
    DISPLAY_VIDEO = False

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=SOURCES[i][0]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating nvtracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    print("Creating nvdsanalytics \n ")
    nvanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
    if not nvanalytics:
        sys.stderr.write(" Unable to create nvanalytics \n")
    nvanalytics.set_property("config-file", "/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps/apps/overspeed_and_anomaly/config_nvdsanalytics.txt")
    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    if DISPLAY_VIDEO:
        print("Creating nvosd \n ")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")
        nvosd.set_property('process-mode',OSD_PROCESS_MODE)
        nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    #sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    # print("Creating EGLSink \n")
    if DISPLAY_VIDEO:
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    sink.set_property('sync',1)
    #sink.set_property('qos',0)
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")
    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', (1/25)*1000*1000)
    pgie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps/apps/overspeed_and_anomaly/config_infer_primary_yoloV8.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('/opt/nvidia/deepstream/deepstream-6.2/sources/deepstream_python_apps/apps/overspeed_and_anomaly/dsnvanalytics_tracker_config.txt')
    config.sections()
    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)
    # mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    # nvvidconv.set_property("nvbuf-memory-type", mem_type)
    # streammux.set_property("nvbuf-memory-type", mem_type)
    # nvvidconv.set_property("nvbuf-memory-type", mem_type)
    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(filter1)
    pipeline.add(nvanalytics)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    if DISPLAY_VIDEO:
        pipeline.add(nvosd)
    # if is_aarch64():
    #     pipeline.add(transform)
    pipeline.add(sink)
    # We link elements in the following order:
    # sourcebin -> streammux -> nvinfer -> nvtracker -> nvdsanalytics ->
    # nvtiler -> nvvideoconvert -> nvdsosd -> sink
    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvanalytics)
    nvanalytics.link(nvvidconv)
    nvvidconv.link(filter1)
    filter1.link(tiler)
    if DISPLAY_VIDEO:
        tiler.link(nvosd)
        nvosd.link(sink)
    else:
        tiler.link(sink)
    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    tiler_src_pad = tiler.get_static_pad("sink")
    if not tiler_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)
        GLib.timeout_add(5000, perf_data.perf_print_callback)
    # List the sources
    print("Now playing...")
    for source in SOURCES.items():
        print("Source: ", source[0], "\ncamera: ", source[1])
    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())
