package com.leetcode2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomizedSet {
	List<Integer> lst;
    Map<Integer, Integer> map;
    /** Initialize your data structure here. */
    public RandomizedSet() {
        lst = new ArrayList<Integer>();
        map = new HashMap<Integer, Integer>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val))
            return false;
        map.put(val, lst.size());
        lst.add(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val))
            return false;
        int index=map.get(val);
        int lastVal = lst.get(lst.size()-1);
        lst.set(index, lastVal);
        lst.remove(lst.size()-1);
        map.put(lastVal, index);
        map.remove(val);
        
        return true;
        
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        int size = lst.size();
        System.out.println(size);
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
    	RandomizedSet obj = new RandomizedSet();
    	boolean param_1 = obj.insert(0);
    	boolean param_2 = obj.remove(0);
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
