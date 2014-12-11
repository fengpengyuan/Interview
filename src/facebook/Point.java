package facebook;

public class Point {
	public double x;

    public double y;
    
    public Point(double x, double y){
    	this.x=x;
    	this.y=y;
    }

    public double distanceFromOrigin(){
            return (this.x*this.x + this.y*this.y); //there is no need add the square root overhead as we only need to compare the distance
    }
    
    public String toString(){
    	return "("+x+", "+y+")";
    }
}
