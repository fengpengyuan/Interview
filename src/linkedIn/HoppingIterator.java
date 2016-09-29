package linkedIn;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class HoppingIterator<T> implements Iterator<T>{
	Iterator<T> it;
	int hop;
	boolean first;
	T nextItem;
	public HoppingIterator(Iterator<T> it, int hop){
		this.it=it;
		this.hop=hop;
		this.first=true;
		this.nextItem=null;
	}
	@Override
	public boolean hasNext() {
		if (nextItem != null) {
			return true;
		}

		if (!first) {
			for (int i = 0; i < hop && it.hasNext(); i++) {
				it.next();
			}
		}

		if (it.hasNext()) {
			nextItem = it.next();
			first = false;
		}

		return nextItem != null;
	}

	@Override
	public T next() {
		T res=nextItem;
		nextItem=null;
		return res;
	}

	@Override
	public void remove() {
		
	}
	
	public static void main(String[] args) {
		List<Integer> list = new ArrayList<Integer>();
		list.add(1);
		list.add(2);
		list.add(3);
		list.add(4);
		list.add(5);

		HoppingIterator<Integer> hi = new HoppingIterator<Integer>(
				list.iterator(), 5);
		while(hi.hasNext()){
			System.out.println(hi.next());
		}
	}

}
