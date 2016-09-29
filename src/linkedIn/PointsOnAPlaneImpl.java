package linkedIn;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;

public class PointsOnAPlaneImpl implements PointsOnAPlane{
	public List<Point> points;
	public PointsOnAPlaneImpl(){
		this.points = new ArrayList<Point>();
	}

	@Override
	public void addPoint(Point point) {
		// TODO Auto-generated method stub
		points.add(point);
	}

	@Override
	public Collection<Point> findNearest(Point center, int p) {
		// TODO Auto-generated method stub
		List<Point> res=new ArrayList<Point>();
		final Point t=center;
		PriorityQueue<Point> heap=new PriorityQueue<Point>(p, new Comparator<Point>(){
			@Override
			public int compare(Point p1, Point p2){
				int dist1=(p1.x-t.x)*(p1.x-t.x)+(p1.y-t.y)*(p1.y-t.y);
				int dist2=(p2.x-t.x)*(p2.x-t.x)+(p2.y-t.y)*(p2.y-t.y);
				return dist2-dist1;
			}
		});
		
		for(int i=0;i<p&&i<points.size();i++){
			heap.offer(points.get(i));
		}
		System.out.println(heap.size());
		for(int i=p;i<points.size();i++){
			Point point=points.get(i);
			int dist=(point.x-t.x)*(point.x-t.x)+(point.y-t.y)*(point.y-t.y);
			Point top=heap.peek();
			int d=(top.x-t.x)*(top.x-t.x)+(top.y-t.y)*(top.y-t.y);
			if(dist<d){
				heap.poll();
				heap.offer(point);
			}
		}
		while(!heap.isEmpty()){
			res.add(heap.poll());
		}
		
		return res;
	}
	
	public int getSize(){
		return points.size();
	}
	
	public static void main(String[] args){
		Point p1=new Point(1,1);
		Point p2=new Point(0,3);
		Point p3=new Point(0,4);
		Point p4=new Point(0,5);
		Point p5=new Point(0,6);
		Point p6=new Point(0,7);
		
		PointsOnAPlaneImpl pop=new PointsOnAPlaneImpl();
		pop.addPoint(p1);
		pop.addPoint(p2);
		pop.addPoint(p3);
		pop.addPoint(p4);
		pop.addPoint(p5);
		pop.addPoint(p6);
		
		Point center=new Point(0,0);
		
		Collection<Point> lst=pop.findNearest(center, 3);
		Iterator<Point> it=lst.iterator();
		System.out.println(lst.size()+" "+pop.getSize());
		while(it.hasNext()){
			Point p=it.next();
			System.out.println("("+p.x+","+p.y+")");
		}
	}

}
