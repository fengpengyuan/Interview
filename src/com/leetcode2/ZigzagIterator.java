package com.leetcode2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ZigzagIterator {
	List<Iterator<Integer>> iterators;
//	int count=0;
	
	 public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
		 iterators = new ArrayList<Iterator<Integer>>();
		 if(v1.size()>0)
			 iterators.add(v1.iterator());
		 if(v2.size()>0)
			 iterators.add(v2.iterator());
	 }
	 
	 public boolean hasNext(){
		 return !iterators.isEmpty();
	 }
	 
	 public int next(){
//		int x = iters.get(count).next();
//		if (!iters.get(count).hasNext())
//			iters.remove(count);
//		else
//			count++;
//
//		if (iters.size() != 0)
//			count %= iters.size();
//		return x;
		 
		 if(!hasNext())
			 return -1;
		 Iterator<Integer> cur=iterators.remove(0);
		 int val=cur.next();
		 if(cur.hasNext())
			 iterators.add(cur);
		 return val;
	 }
}
