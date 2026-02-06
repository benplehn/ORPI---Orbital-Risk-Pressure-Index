/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                cyber: {
                    900: '#050a14', // Deep Space
                    800: '#0a192f',
                    700: '#112240',
                    500: '#00f0ff', // Neon Cyan
                    400: '#00D1FF',
                    300: '#00B8D9',
                    100: '#a6faff',
                },
                risk: {
                    low: '#00ff9d',
                    mid: '#ffbd00',
                    high: '#ff0055',
                }
            },
            fontFamily: {
                mono: ['"Share Tech Mono"', 'monospace'],
                sans: ['"Rajdhani"', 'sans-serif'],
            },
            backgroundImage: {
                'grid-pattern': "linear-gradient(to right, #112240 1px, transparent 1px), linear-gradient(to bottom, #112240 1px, transparent 1px)",
            }
        },
    },
    plugins: [],
}
