"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
import six

if six.PY2:
    from ConfigParser import (
        SafeConfigParser,
        DEFAULTSECT,
        NoSectionError,
        NoOptionError,
    )
else:
    from configparser import (
        SafeConfigParser,
        DEFAULTSECT,
        NoSectionError,
        NoOptionError,
    )

from six.moves import StringIO


class SafeConfigCommentParser (SafeConfigParser, object):
    """
    Adaptation of the ConfigParser.SafeConfigParser object that can also record
    a comment for an option.

    Matched to python 2.6 ConfigParser version at the minimum.

    NOTE: As of right now, comments are not read in from file. They are there
    for creating a configuration object for the intent of writing out to file.
    """

    # Number of characters to limit output printing per line (for comments)
    OUTPUT_LINE_LENGTH = 80

    def __init__(self, defaults=None, dict_type=dict):
        super(SafeConfigCommentParser, self).__init__(defaults, dict_type)

        # We will contain associations of comments to a specific section or
        # option:
        #
        # {
        #   '{section}' : { '__name__' | '{option}' : "comment",
        #                   ... },
        #   ...
        # }
        #
        # If there is no comment associated with an option, there will either
        # be no entry along that path, or that entry will be None.
        #
        self._comments = self._dict()

        # Add slot for DEFAULT section option comments since it won't be added
        # by the user. Also adding an initial null comment to get around bug in
        # comment retrieval.
        self._comments[DEFAULTSECT] = self._dict()
        self.set_comment(None, DEFAULTSECT)

    def write(self, fp):
        """
        Override of write method to provide a better formatted output, taking
        into consideration comments for an option.

        Write an .ini-format representation of the configuration state.

        @type fp: file

        """
        def write_comment(comment):
            """ write comment to file
            @type comment: str

            """
            # Split the comment into blocks based on new-line characters.
            # - Adding a new-line at the end of the comment so that, if an user
            #   explicitly placed a '\n' at the end of their comment, it will be
            #   registered as an expected new comment line with nothing on it.
            blocks = (comment+'\n').splitlines()

            # Each new block with start on a new line, directly underneath the
            # previous block.
            for block in blocks:
                segments = block.split(' ')
                line_buff = ''
                block_buff = ''

                #for seg in segments:
                while segments:
                    # if the concatenation of the current segment would make the
                    # total line buffer length greater than the output length
                    # limit, dump the line to the block buffer.

                    if line_buff and \
                       len(line_buff) + len(segments[0]) + 1 > self.OUTPUT_LINE_LENGTH:
                        # The buffer isn't empty and the next segment won't fit
                        # in the buffer. Dumping out the current buffer into the
                        # block buffer and starting a new line buffer. The '+1'
                        # in the 'if' is to take into account the additional
                        # space to join buffer and the new segment.
                        block_buff += line_buff + '\n'
                        line_buff = ''
                    elif line_buff:
                        # add the segment to the non-empty buffer
                        line_buff = ' '.join([ line_buff, segments.pop(0) ])
                    else:
                        # Nothing in the buffer. Start a new one with the
                        # starting ;
                        line_buff = ' '.join([';', segments.pop(0)])

                # flush the remaining buffer to the block buffer
                if line_buff:
                    block_buff += line_buff + '\n'

                # Write the block out. Every block should already end in a new-
                # line char because of how lines are added to the buffer.
                # - fp is the file object passed to the parent write method
                fp.write( block_buff  )

        if self._defaults:
            fp.write("[%s]\n" % DEFAULTSECT)
            for (key, value) in self._defaults.items():
                # write comment if there is one for this default key
                if ( self._comments.get(DEFAULTSECT, False)
                     and self._comments[DEFAULTSECT].get(key, False) ):
                    write_comment(self._comments[DEFAULTSECT][key])
                # write key/val
                fp.write("%s = %s\n" % (key, str(value).replace('\n', '\n\t')))
            # spacer before next section heading
            fp.write("\n")

        for section in sorted(self._sections):
            fp.write('\n')

            # write comment for section if there is one
            if self.has_comment(section):
                #noinspection PyTypeChecker
                write_comment(self.get_comment(section))

            # section head
            fp.write("[%s]\n" % section)

            for (key, value) in sorted(self._sections[section].items()):
                if key != "__name__":
                    fp.write("\n")
                    # write comment for pair of there is one
                    if self.has_comment(section, key):
                        #noinspection PyTypeChecker
                        write_comment(self.get_comment(section, key))
                    # write key/val
                    fp.write("%s = %s\n" %
                             (key, str(value).replace('\n', '\n\t')))
            fp.write("\n")

    def add_section(self, section, comment=None):
        """
        Create a new section in the configuration with an optional comment.

        Raise DuplicateSectionError if a section by the specified name
        already exists. Raise ValueError if name is DEFAULT or any of it's
        case-insensitive variants.

        @type section:str
        @type comment: str or None

        """
        super(SafeConfigCommentParser, self).add_section(section)

        if comment is not None:
            self._comments[section] = self._dict()
            self._comments[section]['__name__'] = comment

    def set(self, section, option, value=None, comment=None):
        """
        Set an option.  Extend ConfigParser.set: check for string values. Also
        optionally records a comment for the option.

        @type section: str
        @type option: str
        @type value: str or None
        @type comment: str or None

        """
        super(SafeConfigCommentParser, self).set(section, option, value)

        if comment is not None:
            if not section:
                section = DEFAULTSECT

            if section not in self._comments:
                self._comments[section] = self._dict()

            self._comments[section][self.optionxform(option)] = comment

    def has_comment(self, section, option=None):
        """
        Return if the given section heading or option within a section has an
        associated comment.

        @type section: str
        @type option: str or None
        @rtype: bool

        @raise NoSectionError: The provided section does not exist

        """
        if not section:
            section = DEFAULTSECT

        if not self.has_section(section) and section != DEFAULTSECT:
            raise NoSectionError(section)

        if option:
            option = self.optionxform(option)
            if not self.has_option(section, option):
                raise NoOptionError(option, section)
        else:
            # looking for section comment
            option = '__name__'

        return bool(self._comments.get(section, False)
                    and self._comments[section].get(option, False))

    def get_comment(self, section, option=None):
        """
        Get the comment for a section[.option]

        @type section: str
        @param section: Section heading to check for a comment, or the section
        heading that the target option is located under.

        @type option: str or None
        @param option: If checking for an option's comment, this is the option
        under the given section to check. If checking for a section's comment,
        this should be None.

        @rtype: str or None
        @return: The section or option comment if there is one, or None if there
        is no comment for the specified section or option.

        """
        if not section:
            section = DEFAULTSECT

        if not self.has_section(section) and section != DEFAULTSECT:
            raise NoSectionError(section)

        if option:
            option = self.optionxform(option)
            if not self.has_option(section, option):
                raise NoOptionError(option, section)

        else:
            # looking for section comment
            option = '__name__'

        # Combined statement to handle both section and option requests.
        return ( self._comments.get(section, None)
                 and self._comments[section].get(option, None) )

    def set_comment(self, comment, section, option=None):
        """
        Set the comment for a section or option

        @type comment: str or None
        @type section: str
        @type option: str or None

        """
        if not section:
            section = DEFAULTSECT

        if not self.has_section(section) and section != DEFAULTSECT:
            raise NoSectionError(section)

        if section not in self._comments:
            self._comments[section] = self._dict()

        if option:
            option = self.optionxform(option)
            if not self.has_option(section, option):
                raise NoOptionError(option, section)
        else:
            # setting section comment
            option = '__name__'

        self._comments[section][option] = comment

    def remove_comment(self, section, option=None):
        """
        Remove the comment from a section or option.

        @type section: str
        @param section: The section to remove from.

        @type option: str or None
        @param option: The option to remove from or None to remove a section's
        comment.

        @rtype: bool
        @return: True if a comment was removed and False if no comment was
        removed.

        """
        if not section:
            section = DEFAULTSECT
        elif not self.has_section(section) and section != DEFAULTSECT:
            raise NoSectionError(section)

        if option:
            option = self.optionxform(option)
            if not self.has_option(section, option):
                raise NoOptionError(option, section)
        else:
            # setting section comment
            option = '__name__'

        del self._comments[section][option]

    def remove_option(self, section, option):
        """
        Remove an option and any comments associated with it.

        @type section: str
        @param section: The section the option is located in.

        @type option: str
        @param option: The option to remove

        @rtype: bool
        @return: True of the option existed under the given section and was
        removed. False if the option didn't exist and couldn't be removed.

        """
        if super(SafeConfigCommentParser, self).remove_option(section, option):
            if option in self._comments[section]:
                del self._comments[section][option]
            return True
        return False

    def remove_section(self, section):
        """
        Remove a file section and any comments associated with it.

        @type section: str
        @param section: The section to remove.

        @rtype: bool
        @return: True of the section existed and was removed. False if the
        section didn't exist and couldn't be removed.

        """
        if super(SafeConfigCommentParser, self).remove_section(section):
            # With the removal of an entire section entry in the comments
            # dictionary
            # self.remove_comment(section)
            if section in self._comments:
                del self._comments[section]
            return True
        return False

    def as_str(self):
        """
        :return: this configuration as a string as would be written to a file.
        :rtype: str

        """
        buff = StringIO()
        # noinspection PyTypeChecker
        # reason: StringIO implements file-like interface.
        self.write(buff)
        return buff.getvalue()

    __str__ = as_str

    def update(self, *configs):
        """
        Update this configuration object with the values of the given, other
        configuration objects. This updates this object in place. Also updates
        comments.

        The order in which configuration objects are given matters, where the
        first provided will be the first that is updated into this object, and
        the last provided will be the last be the last updated into this object.
        If multiple config objects contain the same section option pairs, the
        value in the last configuration object provided will take priority.

        REMEMBER
        --------
        Section options that are set to an empty string is a valid value and
        will cause an overwrite in this config and/or preceding configs in the
        given order.

        :param configs: One or more SafeConfigCommentParser objects
        :type configs: Iterable of SafeConfigCommentParser
        :return: This config object (in-place update)
        :rtype: SafeConfigCommentParser

        """
        for config in configs:
            self.defaults().update(config.defaults())

            for section in config.sections():
                if not self.has_section(section):
                    self.add_section(section, config.get_comment(section))
                elif config.has_comment(section):
                    self.set_comment(config.get_comment(section), section)

                for option in config.options(section):
                    self.set(section, option, config.get(section, option))
                    # Only update the commend if the current config actually has
                    # one.
                    if config.has_comment(section, option):
                        self.set_comment(config.get_comment(section, option),
                                         section, option)
