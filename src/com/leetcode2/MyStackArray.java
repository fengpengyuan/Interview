package com.leetcode2;

import java.util.Arrays;

public class MyStackArray<E> {
	private final int DEFAULT_CAPACITY=16;
	private int size;
	private Object[] elements;
	
	public MyStackArray(){
		elements=new Object[DEFAULT_CAPACITY];
	}
	
	public void push(E e){
		if(size==elements.length){
			ensureCapa();
		}
		elements[size++] = e;
	}

	public E pop(){
		E ele=(E) elements[--size];
		elements[size]=null;
		return ele;
	}
	
	public void ensureCapa(){
		int newSize=elements.length*2;
		elements=Arrays.copyOf(elements, newSize);
	}
}
