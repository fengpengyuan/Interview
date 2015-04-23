package com.leetcode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;


public class MyDS {
	ArrayList<Integer> list;
	HashMap<Integer, Integer>  map;
	
	public MyDS(){
		list=new ArrayList<Integer>();
		map=new HashMap<Integer, Integer>();
	}
	
	public void insert(int x){
		if(map.containsKey(x))
			return;
		int size=list.size();
		list.add(x);
		map.put(x, size);
	}
	
	public int search(int x){
		return map.get(x);
	}
	
	public void remove(int x){
		if(!map.containsKey(x))
			return;
		
		int index=map.get(x);
		map.remove(x);
		int size=list.size();
		int last=list.get(size-1);
		Collections.swap(list, index, size-1);
		list.remove(size-1);
		map.put(last, index);
	}
	
	public int getRandom(){
		Random rand=new Random();
		int index=rand.nextInt(list.size());
		return list.get(index);
	}
	
	public static void main(String[] args){
		MyDS ds=new MyDS();
		ds.insert(10);
		ds.insert(20);
		ds.insert(30);
		ds.insert(40);
		
		System.out.println(ds.search(30));
        ds.remove(20);
        ds.insert(50);
        System.out.println(ds.search(50));
        System.out.println(ds.getRandom());
	}
}
