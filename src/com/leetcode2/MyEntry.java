package com.leetcode2;

public class MyEntry<K, V> {
	private K key;
	private V val;
	
	public MyEntry(K key, V val){
		this.key=key;
		this.val=val;
	}
	
	public K getKey() {
	    return key;
	  }

	  public V getValue() {
	    return val;
	  }

	  public void setValue(V value) {
	    this.val = value;
	  }
}
