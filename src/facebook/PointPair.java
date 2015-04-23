package facebook;

public class PointPair {
	Point p1;
	Point p2;
	
	public PointPair(Point p1, Point p2){
		this.p1=p1;
		this.p2=p2;
	}
	
	public String toString(){
		return p1.x+", "+p1.y +" and "+ p2.x+", "+p2.y;
	}
}
