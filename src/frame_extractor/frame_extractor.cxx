/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vidl/vidl_config.h>
#include <vidl/vidl_convert.h>
#include <vidl/vidl_frame.h>
#include <vidl/vidl_frame_sptr.h>
#include <vidl/vidl_istream_sptr.h>
#include <vidl/vidl_image_list_istream.h>
#if VIDL_HAS_FFMPEG
#include <vidl/vidl_ffmpeg_istream.h>
#endif
#if VIDL_HAS_DSHOW
#include <vidl/vidl_dshow_file_istream.h>
#endif
#if VIDL_HAS_VIDEODEV2
#include <vidl/vidl_v4l2_device.h>
#include <vidl/vidl_v4l2_istream.h>
#endif
#if VIDL_HAS_VIDEODEV
#include <vidl/vidl_v4l_istream.h>
#endif

#if VIDL_HAS_FFMPEG
#define DEFAULT_ISTREAM_TYPE "ffmpeg"
#else
#define DEFAULT_ISTREAM_TYPE "image_list_glob"
#endif

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>
#include <vil/vil_save.h>

#include <vul/vul_arg.h>
#include <vul/vul_file.h>
#include <vul/vul_sprintf.h>

#include <boost/foreach.hpp>
#include <boost/function.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>

#include <fstream>
#include <iostream>

#define foreach BOOST_FOREACH

static vidl_istream_sptr create_image_list_file_stream(const std::string& param);
static vidl_istream_sptr create_image_list_glob_stream(const std::string& param);
#if VIDL_HAS_FFMPEG
static vidl_istream_sptr create_ffmpeg_stream(const std::string& param);
#endif
#if VIDL_HAS_DSHOW
static vidl_istream_sptr create_dshow_file_stream(const std::string& param);
#endif
#if VIDL_HAS_VIDEODEV2
static vidl_istream_sptr create_v4l2_stream(const std::string& param);
#endif
#if VIDL_HAS_VIDEODEV
static vidl_istream_sptr create_v4l_stream(const std::string& param);
#endif

struct frame_range
{
    frame_range(const std::string& range);

    bool contains(int frame_number);

    typedef std::pair<int, int> subsampling;
    std::vector<subsampling> subsamples;
    std::set<int> frames;
    int min_max;
};

int main(int argc, char* argv[])
{
    vul_arg<std::string> type("-t", "Type of input stream", DEFAULT_ISTREAM_TYPE);
    vul_arg<std::string> file("-i", "File for the input stream");
    vul_arg<std::string> framefile("-ff", "File path to frame numbers", "");
    vul_arg<std::string> frames("-f", "Frame numbers to output", "0-");
    vul_arg<bool> no_basename("-b", "Do not use the basename of the input", false);
    vul_arg<std::string> frame_format("-o", "Output format", "%06d.png");
    vul_arg<bool> one_indexed("-1", "Use one-indexed frame numbers on the output", false);

    vul_arg_parse(argc, argv);

    typedef std::map<std::string, boost::function<vidl_istream_sptr (const std::string&)> > istream_ctor_map_t;
    istream_ctor_map_t istream_types;

    istream_types["image_list_file"] = create_image_list_file_stream;
    istream_types["image_list_glob"] = create_image_list_glob_stream;
#if VIDL_HAS_FFMPEG
    istream_types["ffmpeg"] = create_ffmpeg_stream;
#endif
#if VIDL_HAS_DSHOW
    istream_types["dshow_file"] = create_dshow_file_stream;
#endif
#if VIDL_HAS_VIDEODEV2
    istream_types["v4l2"] = create_v4l2_stream;
#endif
#if VIDL_HAS_VIDEODEV
    istream_types["v4l"] = create_v4l_stream;
#endif

    // Is it a valid stream type?
    istream_ctor_map_t::iterator ctor = istream_types.find(type());
    if (ctor == istream_types.end())
    {
        std::cerr << "Unknown input type: " << type() << std::endl;
        std::cerr << "Known types:";
        foreach (istream_ctor_map_t::value_type& i, istream_types)
        {
            std::cerr << " " << i.first;
        }
        std::cerr << std::endl;
        return 1;
    }

    // Do we have a file?
    if (file() == "")
    {
        std::cerr << "No file given" << std::endl;
        return 1;
    }

    // Create the stream
    vidl_istream_sptr istr = ctor->second(file());

    std::string ffile = framefile();
    std::string str_framerange;
    if (ffile.length() > 0)
    {
        std::ifstream ffinfile(ffile.c_str());
        if (!ffinfile) {
            std::cerr << "Failed to open frame number file \'" << ffile << "\'" << std::endl;
            return 1;
        }

        int tmpi;
        std::vector<int> tmpis;
        std::stringstream framestream;
        while (ffinfile >> tmpi)
        {
            tmpis.push_back(tmpi);
        }
        ffinfile.close();

        for(unsigned int tmpii = 0; tmpii < tmpis.size(); tmpii++)
        {
            if (tmpii == 0)
            {
                framestream << tmpis[tmpii];
            }
            else
            {
                framestream << ',' << tmpis[tmpii];
            }
        }
        str_framerange = framestream.str();
    }
    else if (frames().length() > 0)
    {
        str_framerange = frames();
    }
    else
    {
        std::cerr << "Neither -f nor -ff set" << std::endl;
        return 1;
    }

    frame_range range(str_framerange);
    int idx = 0;
    int frame_number = 0;

    if (one_indexed())
    {
        idx = 1;
        frame_number = 1;
    }

    // Compute the filename for the output format
    std::string file_path = file();

    if (!no_basename())
    {
        file_path = vul_file::basename(file_path);
    }

    std::string fname_safe = file_path;
    // Replace directory separators
    std::replace(fname_safe.begin(), fname_safe.end(), '/', '%');
#ifdef _WIN32
    // Replace backslashes on Windows
    std::replace(fname_safe.begin(), fname_safe.end(), '\\', '%');
    // Replace reserved Windows path characters
    std::string cs = "<>:\"|?*";
    for (size_t c=0; c < cs.length(); c++)
    {
        std::replace(fname_safe.begin(), fname_safe.end(), cs.at(c), '_');
    }
#endif

    // Loop through the video
    while (istr->advance())
    {
        // Check if the frame is wanted
        if (range.contains(frame_number))
        {
            // Grab the frame
            vidl_frame_sptr frame = istr->current_frame();

            // Make the filename
            vul_sprintf frame_name(frame_format().c_str(), frame_number);

            vil_image_view<vxl_byte> image;
            // Try to wrap the frame memory in an image
            if (vidl_pixel_format_color(frame->pixel_format()) == VIDL_PIXEL_COLOR_RGB)
            {
              image = vidl_convert_wrap_in_view(*frame);
            }

            // If unable to wrap, convert the frame to an RGB image
            if (!image)
            {
                vidl_convert_to_view(*frame, image, VIDL_PIXEL_COLOR_RGB);
            }

            if (image)
            {
                // Save the image
                if (!vil_save(image, frame_name))
                {
                    std::cerr << "Failed to save output image \'" << frame_name << "\'" << std::endl;
                    return 1;
                }
            }
            else
            {
                std::cerr << "Unable to convert the image" << std::endl;
            }

            ++idx;
        }

        ++frame_number;
    }

    return 0;
}

vidl_istream_sptr create_image_list_file_stream(const std::string& param)
{
    std::ifstream fin(param.c_str(), std::ios::in);
    std::vector<std::string> paths;

    while (fin.good())
    {
        std::string path;

        std::getline(fin, path);
        paths.push_back(path);
    }

    vidl_image_list_istream* ilstr = new vidl_image_list_istream;
    ilstr->open(paths);

    return vidl_istream_sptr(ilstr);
}

vidl_istream_sptr create_image_list_glob_stream(const std::string& param)
{
    return vidl_istream_sptr(new vidl_image_list_istream(param));
}

#if VIDL_HAS_FFMPEG
vidl_istream_sptr create_ffmpeg_stream(const std::string& param)
{
    return vidl_istream_sptr(new vidl_ffmpeg_istream(param));
}
#endif

#if VIDL_HAS_DSHOW
vidl_istream_sptr create_dshow_file_stream(const std::string& param)
{
    return vidl_istream_sptr(new vidl_dshow_file_istream(param));
}
#endif

#if VIDL_HAS_VIDEODEV2
vidl_istream_sptr create_v4l2_stream(const std::string& param)
{
    vidl_v4l2_device device(param.c_str());

    return vidl_istream_sptr(new vidl_v4l2_istream(device));
}
#endif

#if VIDL_HAS_VIDEODEV
vidl_istream_sptr create_v4l_stream(const std::string& param)
{
    return vidl_istream_sptr(new vidl_v4l_istream(param));
}
#endif

frame_range::frame_range(const std::string& range)
    : min_max(0)
    , subsamples()
{
    int low = 0;
    int cur = 0;
    bool is_multiple = false;

    const size_t sz = range.size();

    for (size_t i = 0; i <= sz; ++i)
    {
        const char c = range.c_str()[i];

        switch (c)
        {
            // A number
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                cur *= 10;
                cur += c - '0';

                break;
            // Range separator
            case '-':
                if (is_multiple)
                {
                    std::cerr << "Warning: Range invalid when subsampling at index " << i << std::endl;

                    low = 0;
                    is_multiple = false;
                }
                // We have a minimum already
                else if (low)
                {
                    std::cerr << "Warning: Triple value range given at index " << i << std::endl;

                    low = 0;
                }
                // Set the minimum
                else if (cur)
                {
                    low = cur;
                }
                // Use 1 as the minimum if there isn't one
                else
                {
                    low = 1;
                }

                cur = 0;

                break;
            // Multiples
            case 'x':
                if (is_multiple)
                {
                    std::cerr << "Warning: Duplicate subsampling step specified at index " << i << std::endl;

                    low = 0;
                    cur = 0;
                    is_multiple = false;
                }
                else if (low || cur)
                {
                    std::cerr << "Warning: Invalid subsampling specified at index " << i << std::endl;

                    low = 0;
                    cur = 0;
                }
                else
                {
                    is_multiple = true;
                }

                break;
            // Offset
            case '+':
                if (!is_multiple || !cur)
                {
                    std::cerr << "Warning: Subsampling needed for an offset at index " << i << std::endl;

                    is_multiple = false;
                }
                else if (low)
                {
                    std::cerr << "Warning: Offset already given for subsampling at index " << i << std::endl;

                    low = 0;
                    is_multiple = false;
                }
                else
                {
                    low = cur;
                    cur = 0;
                }

                break;
            // Separator
            case '\0':
            case ',':
                // in a subsampling state
                if (is_multiple)
                {
                    if (low)
                    {
                        // has subsample as well as offset
                        subsamples.push_back(std::make_pair(low, cur));
                    }
                    else if (cur)
                    {
                        subsamples.push_back(std::make_pair(cur, 0));
                    }
                    else
                    {
                        std::cerr << "Invalid subsampling step given at index " << i << std::endl;
                    }
                }
                // We have a range
                else if (low)
                {
                    // We have a maximum for the range
                    if (cur)
                    {
                        for (int j = low; j <= cur; ++j)
                        {
                            frames.insert(j);
                        }
                    }
                    else
                    {
                        // Lower the boundary if possible
                        if (!min_max || (low < min_max))
                        {
                            min_max = low;
                        }
                    }
                }
                // A single frame
                else if (cur >= 0)
                {
                    frames.insert(cur);
                }
                else
                {
                    std::cerr << "Warning: Empty range given at index " << i << std::endl;
                }

                // Reset
                cur = 0;
                low = 0;
                is_multiple = false;

                break;
            // Invalid character
            default:
                std::cerr << "Warning: Invalid frame range character: \'" << c << "\' at index " << i << std::endl;

                // Reset
                low = 0;
                cur = 0;
                is_multiple = false;

                break;
        }
    }
}

bool frame_range::contains(int frame_number)
{
    // Are we above our minimum?
    if (min_max && (min_max <= frame_number))
    {
        return true;
    }

    // Do we have subsamplings that match
    std::vector<subsampling>::const_iterator i = subsamples.begin();
    std::vector<subsampling>::const_iterator e = subsamples.end();
    for ( ; i != e; ++i)
    {
        if (frame_number >= i->second && ((frame_number - i->second) % i->first) == 0)
        {
            return true;
        }
    }

    // Check if it's in the set
    return (frames.find(frame_number) != frames.end());
}
