//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
var videoPlayer = function(videoElem) {
    var m_videoNode = $(videoElem),
        m_videoElem = videoElem
        m_startTime = null,
        m_endTime = null;

    $(m_videoElem).click(function(event) {
        event.stopPropagation();
    });

    // Public API
    return  {
        videoElement : function() {
            return m_videoElem
        },
        seek : function(time) {
            if (m_videoElem && m_videoElem.readyState === 4) {
                m_videoElem.currentTime = time;
                m_videoElem.pause();
            } else {
                console.log('video is not ready for seek');
            }
        },
        play : function() {
            if (m_videoElem && m_videoElem.readyState === 4) {
                m_videoElem.play();
            } else {
                console.log('video is not ready for play');
            }
        },
        pause : function() {
            if (m_videoElem && m_videoElem.readyState === 4) {
                m_videoElem.pause();
            } else {
                console.log('video is not ready for pause');
            }
        },
        isValid : function(time) {

            try {
                if (m_videoElem && m_videoElem.seekable && m_videoElem.readyState === 4) {
                    m_startTime = m_videoElem.seekable.start(0),
                    m_endTime = m_videoElem.seekable.end(0);

                    if (time >= m_startTime && time <= m_endTime) {
                        return true;
                    }
                    return false;
                }
            } catch (err) {
                console.log(err);
            }

            return true;
        }
    };
};
