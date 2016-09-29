package com.leetcode2;

import java.util.Arrays;

public class MyHashMap<K, V> {
	private int size;
	private final int DEFAULT_CAPACITY=16;
	private MyEntry<K, V>[] values = new MyEntry[DEFAULT_CAPACITY];
	
	public MyHashMap(){
		
	}
	
	public V get(K key){
		for(int i=0;i<size;i++){
			if(values[i]!=null){
				if(values[i].getKey()==key){
					return values[i].getValue();
				}
			}
		}
		return null;
	}
	
	public void put(K key, V v){
		boolean inserted=false;
		for(int i=0;i<size;i++){
			if(values[i].getKey()==key){
				values[i].setValue(v);
				inserted=true;
			}
		}
		if(!inserted){
			ensureCapacity();
			values[size++] = new MyEntry<K, V>(key, v);
		}
	}
	
	public void ensureCapacity(){
		if(size==values.length){
			int newSize=2*values.length;
			values=Arrays.copyOf(values, newSize);
		}
	}
	
	public int size() {
	    return size;
	}
	
	public void remove(K key) {
	    for (int i = 0; i < size; i++) {
	      if (values[i].getKey().equals(key)) {
	        values[i] = null;
	        size--;
	        condenseArray(i);
	      }
	    }
	  }

	private void condenseArray(int start) {
	    for (int i = start; i < size; i++) {
	      values[i] = values[i + 1];
	    }
	  }

}
