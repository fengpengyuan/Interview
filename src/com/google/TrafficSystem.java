package com.google;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class TrafficSystem {
	Queue<Integer> q1, q2, q3, q4;
	Queue<Queue<Integer>> stops;
	int index;
	public TrafficSystem(){
		q1=new LinkedList<Integer>();
		q2=new LinkedList<Integer>();
		q3=new LinkedList<Integer>();
		q4=new LinkedList<Integer>();
		stops = new LinkedList<Queue<Integer>>();
		stops.add(q1);
		stops.add(q2);
		stops.add(q3);
		stops.add(q4);
		index=0;
	}
	public void add(int carId, int roadId){
		if(roadId==1)
			q1.offer(carId);
		if(roadId==2)
			q2.offer(carId);
		if(roadId==3)
			q3.offer(carId);
		if(roadId==4)
			q4.offer(carId);
	}
	
	public int remove(){
		int res = -1;
		int count=0;
		while(stops.peek().isEmpty()){
			stops.add(stops.poll());
			count++;
			if(count==4)
				return res;
		}
		
		
		Queue<Integer> way = stops.poll();
		res = way.poll();
		stops.add(way);
		return res;
	}
	
	
	public static void main(String[] args){
		TrafficSystem ts = new TrafficSystem();
		ts.add(101, 1);
		ts.add(102, 1);
		ts.add(103, 2);
		ts.add(104, 3);
		ts.add(105, 2);
		ts.add(106, 3);
		ts.add(107, 1);
		ts.add(108, 3);
		
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
		System.out.println(ts.remove());
	}
}
