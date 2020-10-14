/**
 * @file Table.h
 * @brief Basic table.
 * @author Thomas - ffyr2w
 * @date 13 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/Traits.h"

namespace Noa {

    /**
     * Simple but flexible table.
     * @details     Store a heap allocated @c std::vector of @c T type.
     * @tparam T    Type to store, also referred as row.
     */
    template<typename T>
    class Table {
    private:
        std::unique_ptr<std::vector<T>> m_table{};

    public:
        /**
         * Default constructor.
         * @param rows      Pre-allocate the vector holding the rows (i.e. @c typename @c T).
         */
        explicit Table(size_t rows = 0) : m_table(std::make_unique<std::vector<T>>(rows)) {}


        /**
         * Add a row to the table.
         * @tparam Args     Should be convertible to @c T
         * @param args      Arguments used to construct an instance of @c T at the end of the table.
         */
        template<typename... Args>
        inline void add(Args&& ... args) {
            m_table->emplace_back(std::forward<Args>(args)...);
        }


        /**
         * Get the data.
         * @return  Reference of the instance's data.
         */
        [[nodiscard]] inline std::vector<T>& data() const noexcept {
            return *m_table;
        }


        /**
         * Apply a @c predicate to every row in the table.
         * @tparam P            A function or lambda that returns void and can take a @c T&
         *                      as first argument.
         * @param predicate     Feed the predicate with each row of type @c T&, sequentially.
         */
        template<typename P,
                typename = std::enable_if_t<std::is_convertible_v<P, std::function<void(T&)>>>>
        inline void apply(P&& predicate) {
            for (auto& row: *m_table)
                predicate(row);
        }


        [[nodiscard]] inline size_t size() const noexcept {
            return m_table->size();
        }


        /**
         * Get a specific row.
         * @param row   Index of the row to return. This is not bound-checked, so one should never
         *              call this function with a @c row that is out of range, since this causes
         *              undefined behavior.
         * @return      Const reference of the @c row.
         */
        [[nodiscard]] inline auto& operator[](size_t row) const {
            return (*m_table)[row];
        }
    };
}
