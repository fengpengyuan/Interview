package com.google;

import java.util.HashSet;
import java.util.Set;

public class Intersection {
	Set<Integer> set =new HashSet<Integer>();
	
	public boolean isIntersection(int[] a, int[] b){
		for(int i: a){
			set.add(i);
		}
		for(int j : b){
			if(set.contains(j))
				return true;
		}
		return false;
	}
	
	public Set<Integer> getIntersection(int[] a, int[] b){
		Set<Integer> res=new HashSet<Integer>();
		for(int i: a){
			set.add(i);
		}
		for(int j: b){
			if(set.contains(j))
				res.add(j);
		}
		return res;
	}
	

	public Set<Integer> getUnion(int[] a, int[] b){
		Set<Integer> res=new HashSet<Integer>();
		for(int i: a){
			set.add(i);
		}
		for(int j: b){
			if(!set.contains(j))
				res.add(j);
		}
		return res;
	}
}
