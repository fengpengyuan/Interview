package linkedIn;

public interface BlockingQueue<T> {
	/** Retrieve and remove the head of the queue, waiting if no elements 
	are present. */
	    T take();

	    /** Add the given element to the end of the queue, waiting if necessary 
	for space to become available. */
	    void put (T obj);
}
