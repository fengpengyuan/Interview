package com.leetcode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Vector2D {
	Iterator<List<Integer>> outIterator;
	Iterator<Integer> inner;
	
	public Vector2D(List<List<Integer>> vec2d){
		outIterator=vec2d.iterator();
	}
	
	public int next(){
		return inner.next();
	}
	
	public boolean hasNext(){
		if(inner!=null&&inner.hasNext())
			return true;
		if(!outIterator.hasNext())
			return false;
		inner=outIterator.next().iterator();
		
		return hasNext();
	}
	
	// make it easier, we can just copy all the elements to one list, and iterate the new list
	
//	List<Integer> iterator;
//	int index;
//	public Vector2D(List<List<Integer>> vec2d){
//		iterator=new ArrayList<Integer>();
//		for(List<Integer> list:vec2d){
//			for(int i: list)
//				iterator.add(i);
//		}
//		index=0;
//	}
//	
//	public int next(){
//		return iterator.get(index++);
//	}
//	
//	public boolean hasNext(){
//		return index<iterator.size();
//	}
	
	public static void main(String[] args){
		List<List<Integer>> lsts=new ArrayList<List<Integer>>();
		List<Integer> l1=new ArrayList<Integer>(Arrays.asList(1,2,3));
		List<Integer> l2=new ArrayList<Integer>(Arrays.asList(4));
		List<Integer> l3=new ArrayList<Integer>(Arrays.asList(5,6));
		List<Integer> l4=new ArrayList<Integer>(Arrays.asList(7));
		
		lsts.add(l1);
		lsts.add(l2);
		lsts.add(l3);
		lsts.add(l4);
		
		Vector2D v=new Vector2D(lsts);
		while(v.hasNext()){
			System.out.println(v.next()+" ");
		}
	}
}
