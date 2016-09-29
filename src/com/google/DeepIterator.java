package com.google;

import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Stack;

public class DeepIterator<T> implements Iterator<T>{
	Stack<Iterator<?>> stk;
	T nextItem;
	
	public DeepIterator(Collection<?> collection) {
		if(collection!=null&&collection.size()!=0){
			stk=new Stack<Iterator<?>>();
			stk.push(collection.iterator());
		}else
			throw new NullPointerException("cannot iterate a null collection");
	}

	@Override
	public boolean hasNext() {
		if(nextItem!=null)
			return true;
		while(!stk.isEmpty()){
			Iterator<?> it=stk.peek();
			if(it.hasNext()){
				Object next=it.next();
				if(next instanceof Collection<?>){
					stk.push(((Collection<?>) next).iterator());
				}else{
					nextItem=(T) next;
					return true;
				}
			}else{
				stk.pop();
			}
		}
		return false;
	}

	@Override
	public T next() {
		if(hasNext()){
			T toReturn=nextItem;
			nextItem=null;
			return toReturn;
		}
		throw new NoSuchElementException("collection is empty!");
	}

	@Override
	public void remove() {
		// TODO Auto-generated method stub
		
	}

}
