"use client";

import { MdSearch } from "react-icons/md";
import styles from "./search.module.css";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useDebouncedCallback } from "use-debounce";

const Search = ({ placeholder }) => {
  const searchParams = useSearchParams();
  const { replace } = useRouter();
  const pathname = usePathname();

  const handleSearch = useDebouncedCallback((e) => {
    const params = new URLSearchParams(searchParams);

    params.set("page", 1);

    if (e.target.value) {
      e.target.value.length > 2 && params.set("q", e.target.value);
    } else {
      params.delete("q");
    }
    replace(`${pathname}?${params}`);
  }, 300);

  const clearSearch = () => {
    const params = new URLSearchParams(searchParams);
    params.delete("q");
    params.set("page", 1);
    replace(`${pathname}?${params}`);
    document.querySelector(`.${styles.input}`).value = "";
  };

  const isSearchActive = searchParams.get("q");

  return (
    <div className={`${styles.container} ${isSearchActive ? styles.active : ""}`}>
      <MdSearch className={styles.searchIcon} />
      <input
        type="text"
        placeholder={placeholder}
        className={styles.input}
        onChange={handleSearch}
        defaultValue={searchParams.get("q") || ""}
      />
      {isSearchActive && (
        <button 
          onClick={clearSearch} 
          className={styles.clearButton}
          aria-label="Clear search"
        >
          ×
        </button>
      )}
    </div>
  );
};

export default Search;
