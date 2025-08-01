import { NextResponse } from 'next/server';

export function middleware(request) {
  // Get the pathname
  const pathname = request.nextUrl.pathname;
  
  // Check if the path should be protected (starts with /dashboard)
  const isProtectedRoute = pathname.startsWith('/dashboard');
  
  // Skip middleware for public assets and API routes
  if (
    pathname.startsWith('/_next') || 
    pathname.startsWith('/api') || 
    pathname.startsWith('/static') || 
    pathname.includes('.') ||
    pathname === '/login' ||
    pathname === '/logout'
  ) {
    return NextResponse.next();
  }
  
  // Get the token from cookies
  const token = request.cookies.get('auth-token')?.value;
  
  // If there's no token and it's a protected route, redirect to login
  if (isProtectedRoute && !token) {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  
  // Allow access to the route
  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|static|.*\\..*|_next).*)'],
};
