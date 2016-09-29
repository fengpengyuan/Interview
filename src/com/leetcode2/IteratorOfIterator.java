package com.leetcode2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class IteratorOfIterator {
	List<Iterator<Integer>> iterators;
	
	public IteratorOfIterator(List<List<Integer>> lists){
		iterators=new ArrayList<Iterator<Integer>>();
		for(List<Integer> lst: lists){
			if(lst.size()>0){
				iterators.add(lst.iterator());
			}
		}
	}
	
	public boolean hasNext() {
		return !iterators.isEmpty();
	}
	
	public int next() {
		if(!hasNext())
			return -1;
		Iterator<Integer> cur=iterators.remove(0);
		int val=cur.next();
		if(cur.hasNext())
			iterators.add(cur);
		return val;
	}
	
	public static void main(String[] args){
		List<List<Integer>> lists=new ArrayList<List<Integer>>();
		List<Integer> l1=new ArrayList<Integer>(Arrays.asList(1,2,3,4));
		List<Integer> l2=new ArrayList<Integer>(Arrays.asList(5,6));
		List<Integer> l3=new ArrayList<Integer>(Arrays.asList(7,8,9));
		List<Integer> l4=new ArrayList<Integer>(Arrays.asList(10));
		lists.add(l1);
		lists.add(l2);
		lists.add(l3);
		lists.add(l4);
		
		IteratorOfIterator it=new IteratorOfIterator(lists);
		while(it.hasNext()){
			System.out.print(it.next()+" ");
		}
	}
}
