package com.leetcode2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Vector2D {
	Iterator<List<Integer>> it;
	Iterator<Integer> cur;
	public Vector2D(List<List<Integer>> vector){
		it=vector.iterator();
	}
	
	public int next(){
		hasNext();
		return cur.next();
	}
	
	public boolean hasNext(){
		while(cur==null||!cur.hasNext()&&it.hasNext()){
			cur=it.next().iterator();
		}
		return cur!=null&&cur.hasNext();
	}
	
	public void remove(){
//		next();
		cur.remove();
	}
	
	public static void main(String[] args){
		List<List<Integer>> lsts=new ArrayList<List<Integer>>();
		List<Integer> l1=new ArrayList<Integer>(Arrays.asList(1,2));
		List<Integer> l2=new ArrayList<Integer>(Arrays.asList(3));
//		List<Integer> l3=new ArrayList<Integer>(Arrays.asList(5,6));
//		List<Integer> l4=new ArrayList<Integer>(Arrays.asList(7));
		
		lsts.add(l1);
		lsts.add(l2);
//		lsts.add(l3);
//		lsts.add(l4);
		
//		hasnext() getnext() hasnext() getnext() remove()
		
		Vector2D v=new Vector2D(lsts);
		System.out.println(v.hasNext());
		System.out.println(v.next());
		System.out.println(v.hasNext());
		System.out.println(v.next());
		v.remove();
		System.out.println(lsts);
//		while(v.hasNext()){
//			System.out.println(v.next()+" ");
//		}
	}
}
