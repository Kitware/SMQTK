# coding=utf-8


class SimpleBoundingBox (object):

    def __init__(self, p1, p2):
        self.min_x = min(p1[0], p2[0])
        self.min_y = min(p1[1], p2[1])
        self.max_x = max(p1[0], p2[0])
        self.max_y = max(p1[1], p2[1])

        # # Maybe more efficient? requires testing
        # pts = (p1, p2)
        # x_pt_min = (min(p1[0], p2[0]) == p2[0])
        # y_pt_min = (min(p1[1], p2[1]) == p2[1])
        # self.min_x = pts[x_pt_min][0]
        # self.max_x = pts[not x_pt_min][0]
        # self.min_y = pts[y_pt_min][1]
        # self.max_y = pts[not y_pt_min][1]

    def __repr__(self):
        return "SimpleBoundingBox{(%f, %f), (%f, %f)}" \
            % (self.min_x, self.min_y, self.max_x, self.max_y)

    def contains_point(self, x, y):
        """ Test if a point lies in the box

        True if edge hit

        """
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)
