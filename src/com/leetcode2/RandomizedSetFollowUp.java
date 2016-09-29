package com.leetcode2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

//The follow-up: allowing duplications.
public class RandomizedSetFollowUp {
	List<Integer> lst;
    Map<Integer, Set<Integer>> map;
    /** Initialize your data structure here. */
    public RandomizedSetFollowUp() {
        lst = new ArrayList<Integer>();
        map = new HashMap<Integer, Set<Integer>>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(!map.containsKey(val))
            map.put(val, new HashSet<Integer>());
        map.get(val).add(lst.size());
        lst.add(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
//        if(!map.containsKey(val))
//            return false;
//        int index=map.get(val).iterator().next();
//        map.get(val).remove(index);
//        int lastVal = lst.get(lst.size()-1);
//        lst.set(index, lastVal);
//        map.get(lastVal).remove(lst.size()-1);
//        map.get(lastVal).add(index);
//        lst.remove(lst.size()-1);
//        if(map.get(val).isEmpty())
//        	map.remove(val);
//        return true;
    	
    	boolean contain = map.containsKey(val);
        if ( ! contain ) return false;
        int loc = map.get(val).iterator().next();
            map.get(val).remove(loc);
        if (loc < lst.size() - 1 ) {
            int lastone = lst.get(lst.size() - 1 );
            lst.set( loc , lastone );
            map.get(lastone).remove(lst.size() - 1);
            map.get(lastone).add(loc);
        }
        lst.remove(lst.size() - 1);
        if (map.get(val).isEmpty()) map.remove(val);
        return true;
    	
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        int size = lst.size();
        int index = new Random().nextInt(size);
        return lst.get(index);
    }
    
    public static void main(String[] args){
    	/**
    	 * Your RandomizedSet object will be instantiated and called as such:
    	 * RandomizedSet obj = new RandomizedSet();
    	 * boolean param_1 = obj.insert(val);
    	 * boolean param_2 = obj.remove(val);
    	 * int param_3 = obj.getRandom();
    	 */
    	RandomizedSetFollowUp obj = new RandomizedSetFollowUp();
    	boolean param_1 = obj.insert(0);
    	param_1 = obj.insert(0);
    	param_1 = obj.insert(0);
    	boolean param_2 = obj.remove(0);
    	param_2 = obj.remove(0);
    	param_2 = obj.remove(0);
    	param_1 = obj.insert(-1);
    	param_2 = obj.remove(-1);
    	int param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    	param_3 = obj.getRandom();
    }
}
