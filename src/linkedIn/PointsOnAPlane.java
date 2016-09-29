package linkedIn;

import java.util.Collection;

public interface PointsOnAPlane {
	/** 
     * Stores a given point in an internal data structure
     */
    void addPoint(Point point);

    /**
     * For given 'center' point returns a subset of 'p' stored points.
     * that are closer to the center than others.. 
     *
     * E.g.
     * Stored:
     * (1, 1)
     * (0, 3)
     * (0, 4)
     * (0, 5)
     * (0, 6)
     * (0, 7) 
     *
     * findNearest(new Point(0, 0), 3) -> (1, 1), (0, 3), (0, 4)
     */
    
    Collection<Point> findNearest(Point center, int p);

    static class Point {
        final int x;
        final int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        } 
    }
}
