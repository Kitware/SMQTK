Frame extraction tool

Motivation:

    This tool should be used to extract frames from videos. This ensures a
    consistent naming scheme and allows lists of frame numbers to be
    consistent with each other. Frames are one-indexed.

Input:

    Videos can be input with the "-i" flag. By default, ffmpeg is used if
    available, otherwise a glob for images is accepted. The input type can be
    specified with the "-t" parameter. A list of supported types is as follows:

        image_list_glob

            The "-i" flag is a glob (similar to shell globbing) that matches
            the desired images (alphabetical order).

        image_list_file

            The "-i" flag is a file with a list of images per line.

        ffmpeg

            The "-i" flag is a video file in a format supported by ffmpeg.

    Windows only:

        dshow_file

            The "-i" flag is a video file in a format supported by the
            DirectShow API.

    Linux only:

        v4l2

            The "-i" flag is a video file in a format supported by the
            Video4Linux2 API.

        v4l

            The "-i" flag is a video file in a format supported by the
            Video4Linux API.

Output:

    Frames can be selected from the input by using the "-f" flag. This
    parameter is a list of the frame numbers to output. The default is "0-".
    The following formats are supported:

        FRAME

            A decimal integer > 0.

            Includes the frame with the index to be included in the output.

            Example:

                "12"

                    Includes frame 12 in the output.

        RANGE

            ${FRAME}-${FRAME}

            Includes the frames with an index between or equal to the
            endpoints of the given indexes in the output.

            Example:

                "10-14"

                    Includes frames 10, 11, 12, 13, and 14 in the output.

        MINRANGE

            ${FRAME}-

            Includes all frames with an index of at least the given frame
            number in the output.

            Example:

                "10-"

                    Includes frame 10 as well as subsequent frames in the
                    output.

        MAXRANGE

            -${FRAME}

            Includes all frames up to and including the given frame number in
            the output.

            Example:

                "-5"

                    Includes frames 1, 2, 3, 4, and 5 in the output.

        SUBSAMPLING

            x${FRAME}

            Includes all frames that are a multiple of the given number in the
            output.

            Example:

                "x2"

                    Includes all frames that have an index that is a multiple
                    of 2 (i.e., 2, 4, 6, 8, etc.).

        SUBSAMPLING_WITH_OFFSET

            ${SUBSAMPLING}+${FRAME}

            Includes all frames that are a multiple of the given subsampling,
            but offset by the given frame number. The frame offset does not
            have to be less than the subsampling, allowing the specification
            of an arbitrary start frame for the offset.

            Example:

                "x2+1"

                    Includes all frames that have an index that is 1 more than
                    a multiple of 2 (i.e., 1, 3, 5, 7, etc.).

                "x5+12"

                    Includes all frames that have an index that is 12 more
                    than a multiple of 5 (i.e. 12, 17, 22, 27, 32, etc.).

    These formats can be combined with a ",". Invalid formats are ignored
    after outputting a warning message. Frames may also be specified in a file
    given by the "-ff" flag. A "," is automatically inserted between lines.

    Examples:

        "-5,10-15,x6+3,30-"

            Includes frames 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 21, 27,
            30, and all subsequent frames in the output.

        "-5,x10+5"

            Includes frames 1, 2, 3, 4, 5, 15, 25, 35, etc. in the output.
