"use client";

import styles from "./pagination.module.css";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { MdNavigateBefore, MdNavigateNext, MdFirstPage, MdLastPage } from "react-icons/md";

const Pagination = ({ count }) => {
  const searchParams = useSearchParams();
  const { replace } = useRouter();
  const pathname = usePathname();

  const page = parseInt(searchParams.get("page") || 1);
  const params = new URLSearchParams(searchParams);
  const ITEM_PER_PAGE = 8;

  // Calculate total pages
  const totalPages = Math.ceil(count / ITEM_PER_PAGE);
  
  const hasPrev = page > 1;
  const hasNext = page < totalPages;

  const handleChangePage = (newPage) => {
    params.set("page", newPage);
    replace(`${pathname}?${params}`);
  };

  return (
    <div className={styles.container}>
      <div className={styles.pageInfo}>
        Showing page <span className={styles.currentPage}>{page}</span> of 
        <span className={styles.totalPages}>{totalPages}</span>
      </div>
      
      <div className={styles.buttonsContainer}>
        <button
          className={styles.button}
          disabled={!hasPrev}
          onClick={() => handleChangePage(1)}
          title="First Page"
        >
          <MdFirstPage />
          First
        </button>
        
        <button
          className={styles.button}
          disabled={!hasPrev}
          onClick={() => handleChangePage(page - 1)}
          title="Previous Page"
        >
          <MdNavigateBefore />
          Prev
        </button>
        
        <button
          className={styles.button}
          disabled={!hasNext}
          onClick={() => handleChangePage(page + 1)}
          title="Next Page"
        >
          Next
          <MdNavigateNext />
        </button>
        
        <button
          className={styles.button}
          disabled={!hasNext}
          onClick={() => handleChangePage(totalPages)}
          title="Last Page"
        >
          Last
          <MdLastPage />
        </button>
      </div>
    </div>
  );
};

export default Pagination;
